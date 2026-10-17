"""Local desktop interface. All processing stays in a background worker thread."""
import json
import os
from pathlib import Path
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from patient_pipeline.runtime import Control, StopRequested
from run_project import run_project
from patient_pipeline.project_model import ProjectOptions
from patient_pipeline.desktop_runtime import configure_tk

configure_tk()

PROJECT=Path(__file__).resolve().parent
PHASES={'discover':('查找资料',0,5),'headers':('检查文件头',5,25),
        'documents':('关联病例',25,40),'images':('处理影像',40,95),'reports':('保存报告',95,100)}


class App:
    def __init__(self,root,*,repair=False,owner=None):
        self.root=root;self.thread=None;self.control=None;self.events=queue.Queue()
        self.is_repair=repair;self.owner=owner;self.repair_window=None;self.active_panel=None
        self.poll_id=None
        self.closing=False;self.started=0;self.last_message='';self.last_log=0
        title='AAOCA 数据修复' if repair else 'AAOCA 资料整理与增量项目'
        root.title(title);root.geometry('1040x850');root.minsize(900,820 if repair else 760)
        root.configure(bg='#f2f5f8')
        style=ttk.Style(root);style.theme_use('clam')
        style.configure('TFrame',background='#f2f5f8');style.configure('TLabel',background='#f2f5f8',font=('Microsoft YaHei UI',10))
        style.configure('Title.TLabel',font=('Microsoft YaHei UI',20,'bold'),foreground='#12354b')
        style.configure('TButton',font=('Microsoft YaHei UI',10),padding=(14,8))
        style.configure('Accent.TButton',background='#146c94',foreground='white')
        style.configure('TProgressbar',background='#168b98',troughcolor='#dbe6ee',thickness=16)
        frame=ttk.Frame(root,padding=24);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text=title,style='Title.TLabel').pack(anchor='w')
        introduction=('选择修复方法后执行；不勾选时按现有规则重试。完成后自动更新分类和报告。' if repair
                      else 'CT、超声、PDF 或主表可任意组合。后续添加新资料，继续使用同一个结果目录。')
        ttk.Label(frame,text=introduction).pack(anchor='w',pady=(8,14))
        self.source=tk.StringVar();self.output=tk.StringVar();self.patients=tk.StringVar()
        self.audit=tk.BooleanVar();self.force=tk.BooleanVar()
        self.mode=tk.StringVar(value='添加资料 / 继续处理')
        self.match_names=tk.BooleanVar();self.recover_copies=tk.BooleanVar();self.split_phases=tk.BooleanVar()
        self.inputs=[]
        source_label=('补充资料（可留空，使用已登记的源路径；多个路径用 | 分隔）' if repair
                      else '本次资料（多个路径用 | 分隔；追加时只填新资料）')
        for title,var,choose in [(source_label,self.source,self.pick_source),('项目结果目录（新项目用空目录；已有结果可直接接纳）',self.output,self.pick_output)]:
            ttk.Label(frame,text=title).pack(anchor='w')
            row=ttk.Frame(frame);row.pack(fill='x',pady=(4,12))
            entry=ttk.Entry(row,textvariable=var,font=('Microsoft YaHei UI',10));entry.pack(side='left',fill='x',expand=True)
            button=ttk.Button(row,text='浏览…',command=choose);button.pack(side='right',padx=(10,0))
            self.inputs.extend([entry,button])
        if repair:
            options=ttk.Frame(frame);options.pack(fill='x',pady=(4,8))
            ttk.Label(options,text='修复方法').pack(side='left',padx=(0,10))
            for label,var in [('唯一姓名匹配',self.match_names),('完整副本选择',self.recover_copies),('明确期相拆分',self.split_phases)]:
                b=ttk.Checkbutton(options,text=label,variable=var);b.pack(side='left',padx=(0,20));self.inputs.append(b)
            ttk.Label(frame,text='姓名须唯一对应；副本须验证完整；期相须有明确依据。原始资料不改写。').pack(anchor='w',pady=(0,6))
        else:
            options=ttk.Frame(frame);options.pack(fill='x',pady=(0,8))
            ttk.Label(options,text='本次操作').pack(side='left',padx=(0,10))
            combo=ttk.Combobox(options,textvariable=self.mode,values=['添加资料 / 继续处理','仅更新报告（无需原资料）','只检查，不转换影像'],state='readonly',width=29)
            combo.pack(side='left');self.inputs.append(combo)
        row=ttk.Frame(frame);row.pack(fill='x',pady=(0,12))
        ttk.Label(row,text='限定转换编号（可选；报告仍保留全项目）').pack(side='left')
        ids=ttk.Entry(row,textvariable=self.patients);ids.pack(side='left',fill='x',expand=True,padx=(10,0));self.inputs.append(ids)
        if repair:
            ttk.Label(frame,text='编号仅限制影像和文档处理；姓名关联会核对全项目。此前已确认的修复继续有效。',wraplength=850).pack(anchor='w',pady=(0,10))
        actions=ttk.Frame(frame);actions.pack(fill='x',pady=(2,16))
        self.start_button=ttk.Button(actions,text='执行修复 / 重试' if repair else '开始 / 续跑',style='Accent.TButton',command=self.start);self.start_button.pack(side='left')
        if not repair:
            self.repair_button=ttk.Button(actions,text='修复数据…',command=self.open_repair)
            self.repair_button.pack(side='left',padx=(10,0));self.inputs.append(self.repair_button)
        self.pause_button=ttk.Button(actions,text='暂停',command=self.toggle_pause,state='disabled');self.pause_button.pack(side='left',padx=10)
        self.stop_button=ttk.Button(actions,text='安全停止',command=self.stop,state='disabled');self.stop_button.pack(side='left')
        self.open_button=ttk.Button(actions,text='查看结果与问题' if repair else '打开结果',command=self.open_results);self.open_button.pack(side='right')
        self.status=tk.StringVar(value='准备就绪');self.phase=tk.StringVar(value='尚未开始');self.stats=tk.StringVar(value='')
        ttk.Label(frame,textvariable=self.status,font=('Microsoft YaHei UI',12,'bold')).pack(anchor='w')
        self.overall=ttk.Progressbar(frame,maximum=100);self.overall.pack(fill='x',pady=(10,6))
        self.detail=ttk.Progressbar(frame,maximum=100);self.detail.pack(fill='x',pady=(0,6))
        ttk.Label(frame,textvariable=self.phase).pack(anchor='w')
        ttk.Label(frame,textvariable=self.stats).pack(anchor='w',pady=(6,10))
        self.result=tk.StringVar(value='尚未执行修复；执行后在这里显示全项目报告前后对比。' if repair else '')
        if repair:
            ttk.Label(frame,textvariable=self.result,wraplength=900,justify='left').pack(anchor='w',pady=(0,10))
        self.log=tk.Text(frame,height=8,wrap='word',font=('Microsoft YaHei UI',9),background='white',relief='flat',padx=10,pady=8,state='disabled')
        self.log.pack(fill='both',expand=True)
        ttk.Label(frame,text='暂停/停止在安全检查点生效；当前解码或写入可能需要先完成。原始资料始终保留。',font=('Microsoft YaHei UI',9),wraplength=850).pack(side='bottom',anchor='w',pady=(10,0),before=self.log)
        self.settings=(owner.settings.parent/'repair_interface_settings.json') if owner else PROJECT/'back/interface_settings.json'
        try:
            if owner:
                self.output.set(owner.output.get());self.patients.set(owner.patients.get())
            else:
                values=json.loads(self.settings.read_text(encoding='utf-8'))
                self.source.set(values.get('input',''));self.output.set(values.get('output',''));self.patients.set(values.get('patients',''))
        except (OSError,ValueError):pass
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.bind('<Destroy>',self.on_destroy,add='+')
        self.poll_id=root.after(100,self.poll)
    def open_repair(self):
        if self.active_panel or self.closing:return
        if self.repair_window is None:
            window=tk.Toplevel(self.root)
            self.repair_window=App(window,repair=True,owner=self)
        self.repair_window.root.deiconify();self.repair_window.root.lift()
    def set_busy(self,busy):
        coordinator=self.owner or self
        for panel in [coordinator,coordinator.repair_window]:
            if panel is None:continue
            for widget in panel.inputs+[panel.start_button]:
                widget.configure(state='disabled' if busy else ('readonly' if isinstance(widget,ttk.Combobox) else 'normal'))
    def on_destroy(self,event):
        if event.widget!=self.root:return
        if self.poll_id:
            self.root.after_cancel(self.poll_id);self.poll_id=None
        if self.owner and self.owner.repair_window is self:self.owner.repair_window=None
    def pick_source(self):
        value=filedialog.askdirectory(title='选择本批次资料目录；可直接在输入栏添加其他路径')
        if value:self.source.set(value)
    def pick_output(self):
        value=filedialog.askdirectory(title='选择空文件夹或之前的结果文件夹；也可在输入框填写新路径',initialdir=self.output.get() or None)
        if value:self.output.set(value)
    def append_log(self,text):
        self.log.configure(state='normal');self.log.insert('end',time.strftime('%H:%M:%S')+'  '+text+'\n')
        if int(self.log.index('end-1c').split('.')[0])>160:self.log.delete('1.0','40.0')
        self.log.see('end');self.log.configure(state='disabled')
    def start(self):
        coordinator=self.owner or self
        if coordinator.active_panel or coordinator.closing or self.closing:return
        source=self.source.get().strip().strip('"');output=self.output.get().strip().strip('"')
        action='retry' if self.is_repair else {'添加资料 / 继续处理':'append','仅更新报告（无需原资料）':'report','只检查，不转换影像':'audit'}.get(self.mode.get(),'append')
        if not output or (not source and action in {'append','audit'}):messagebox.showinfo('请填写路径','请选择本次资料和项目结果目录。仅更新报告、重试失败可不填资料路径。');return
        patients=[x for x in re.split(r'[\s,，;；]+',self.patients.get().strip()) if x]
        options=ProjectOptions(action,self.is_repair and self.match_names.get(),self.is_repair and self.recover_copies.get(),self.is_repair and self.split_phases.get(),tuple(patients))
        args=(source,output,options)
        self.control=Control(self.events.put);self.started=time.monotonic();self.last_message=''
        self.overall['value']=0;self.detail.stop();self.detail['value']=0;self.status.set('正在开始…')
        self.stats.set('');self.phase.set('正在准备…')
        if self.is_repair:
            selected=[name for name,enabled in [('唯一姓名匹配',options.match_names),('完整副本选择',options.recover_copies),('明确期相拆分',options.split_phases)] if enabled]
            self.result.set('本次：'+('、'.join(selected) if selected else '按现有规则重试，不启用新修复')+'；执行结束后更新统计。')
            self.append_log(self.result.get())
        coordinator.active_panel=self;self.set_busy(True)
        self.pause_button.configure(state='normal',text='暂停');self.stop_button.configure(state='normal')
        try:
            self.settings.parent.mkdir(exist_ok=True)
            self.settings.write_text(json.dumps({'input':source,'output':output,'patients':self.patients.get()},ensure_ascii=False),encoding='utf-8')
        except OSError:pass
        self.thread=threading.Thread(target=self.work,args=args,daemon=False)
        try:self.thread.start()
        except Exception as exc:self.handle({'finished':True,'code':1,'message':f'无法启动任务：{exc}'})
    def work(self,source,output,options):
        try:
            before=self.read_overview(output) if self.is_repair else None
            sources=[] if options.action=='report' else [p.strip().strip('"') for p in source.split('|') if p.strip()]
            code=run_project(sources,output,options,self.control)
            event={'finished':True,'code':code,'message':'处理完成，请查看问题清单。' if code==2 else '全部处理完成。'}
            if self.is_repair:event['repair_result']=self.compare_overview(before,self.read_overview(output))
            self.events.put(event)
        except StopRequested:self.events.put({'finished':True,'code':130,'message':'已安全停止。再次点击开始，可从已保存的进度继续。'})
        except BaseException as exc:self.events.put({'finished':True,'code':1,'message':f'任务已停止：{exc}\n修复问题后可使用相同路径续跑。'})
    @staticmethod
    def read_overview(output):
        try:
            data=json.loads((Path(output)/'项目数据快照.json').read_text(encoding='utf-8-sig'))
            overview=data.get('overview')
            return overview if isinstance(overview,dict) else None
        except (OSError,ValueError,AttributeError):return None
    @staticmethod
    def compare_overview(before,after):
        if not after:return '处理已结束，但无法读取报告统计；请点击“查看结果与问题”检查。'
        lines=['全项目报告前后对比（不是本次修复成功项数；CT 按原始序列，超声/PDF 按处理单元）：']
        for key,label in [('CT','CT'),('US','超声'),('PDF','PDF')]:
            old=(before or {}).get(key,{});new=after.get(key,{})
            lines.append(f'{label}：成功 {old.get("success","—")} → {new.get("success",0)}；失败 {old.get("failed","—")} → {new.get("failed",0)}；待关联 {old.get("pending","—")} → {new.get("pending",0)}')
        return '\n'.join(lines)
    def toggle_pause(self):
        if self.control:
            if self.control.gate.is_set():self.control.pause();self.pause_button.configure(text='继续')
            else:self.control.resume();self.pause_button.configure(text='暂停')
    def stop(self):
        if self.control:self.control.stop();self.pause_button.configure(state='disabled');self.stop_button.configure(state='disabled')
    def poll(self):
        try:
            while True:self.handle(self.events.get_nowait())
        except queue.Empty:pass
        coordinator=self.owner or self
        if self.closing and (coordinator.active_panel is not self if self.owner else coordinator.active_panel is None):
            self.root.destroy();return
        self.poll_id=self.root.after(100,self.poll)
    def handle(self,event):
        message=event.get('message','');state=event.get('state')
        if message and (message!=self.last_message) and (time.monotonic()-self.last_log>1 or event.get('level') or event.get('finished') or state in {'paused','pausing','stopping'}):
            self.append_log(message);self.last_message=message;self.last_log=time.monotonic()
        if state in {'paused','pausing','stopping'}:self.status.set(message)
        if event.get('stage') and not (self.control and (not self.control.gate.is_set() or self.control.stopping.is_set())):
            name,low,high=PHASES[event['stage']];done=event.get('done',0);total=event.get('total',0)
            self.status.set(name);elapsed=int(time.monotonic()-self.started)
            if total:
                self.detail.stop();self.detail.configure(mode='determinate');fraction=min(1,max(0,done/total))
                self.detail['value']=fraction*100;self.overall['value']=low+(high-low)*fraction
                self.phase.set(f'{name}：{int(done)} / {total}  ·  阶段 {fraction:.0%}  ·  已用时 {elapsed//60}分{elapsed%60}秒（含暂停）')
            else:
                if str(self.detail['mode'])!='indeterminate':self.detail.configure(mode='indeterminate');self.detail.start(15)
                self.phase.set(name+'…')
        if 'counts' in event:
            c=event['counts']
            if self.is_repair:
                self.stats.set(f'本次已处理 {c.get("units_processed",0)} 个单元  ·  影像复用 {c.get("images_reused",0)} 项  ·  文档复用 {c.get("documents_reused",0)} 份')
            else:self.stats.set(f'本次转换 {c.get("images_converted",0)} 项  ·  已跳过 {c.get("images_reused",0)+c.get("images_adopted",0)} 项  ·  病例复用 {c.get("documents_reused",0)} 份')
        if event.get('finished'):
            self.detail.stop();self.status.set({0:'全部处理完成',2:'处理完成，请查看问题清单',130:'已安全停止，可以续跑',1:'任务已停止，请查看下方原因，修复后续跑'}.get(event['code'],message))
            if event['code'] in (0,2):self.overall['value']=100;self.detail['value']=100
            if self.is_repair:
                self.status.set({0:'修复 / 重试流程完成',2:'修复 / 重试完成，项目仍有问题，请查看结果',130:'已安全停止，可在本窗口继续重试',1:'修复任务停止，请查看下方原因'}.get(event['code'],message))
                self.result.set(event.get('repair_result','本次未完成，未刷新统计；已完成的处理记录保留。'))
                self.append_log(self.result.get())
            (self.owner or self).active_panel=None;self.set_busy(False)
            self.pause_button.configure(state='disabled',text='暂停');self.stop_button.configure(state='disabled')
    def open_results(self):
        path=Path(self.output.get().strip().strip('"'))
        if (path/'查看结果.html').is_file():os.startfile(path/'查看结果.html')
        elif path.is_dir():os.startfile(path)
        else:messagebox.showinfo('尚无结果','先选择输出文件夹并运行任务。')
    def close(self):
        coordinator=self.owner or self
        active=coordinator.active_panel
        if active and (self.owner is None or active is self):
            self.closing=True;active.stop()
        else:self.root.destroy()


def main():
    root=tk.Tk();App(root);root.mainloop()


if __name__=='__main__':main()
