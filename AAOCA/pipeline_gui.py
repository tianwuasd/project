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
from run_pipeline import run

PROJECT=Path(__file__).resolve().parent
PHASES={'discover':('查找资料',0,5),'headers':('检查文件头',5,25),
        'documents':('关联病例',25,40),'images':('处理影像',40,95),'reports':('保存报告',95,100)}


class App:
    def __init__(self,root):
        self.root=root;self.thread=None;self.control=None;self.events=queue.Queue()
        self.closing=False;self.started=0;self.last_message='';self.last_log=0
        root.title('病例与影像整理');root.geometry('920x720');root.minsize(760,620)
        root.configure(bg='#f2f5f8')
        style=ttk.Style(root);style.theme_use('clam')
        style.configure('TFrame',background='#f2f5f8');style.configure('TLabel',background='#f2f5f8',font=('Microsoft YaHei UI',10))
        style.configure('Title.TLabel',font=('Microsoft YaHei UI',20,'bold'),foreground='#12354b')
        style.configure('TButton',font=('Microsoft YaHei UI',10),padding=(14,8))
        style.configure('Accent.TButton',background='#146c94',foreground='white')
        style.configure('TProgressbar',background='#168b98',troughcolor='#dbe6ee',thickness=16)
        frame=ttk.Frame(root,padding=24);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='病例与影像整理',style='Title.TLabel').pack(anchor='w')
        ttk.Label(frame,text='选择两个文件夹，即可统计、检查并转换。再次选择相同路径，会自动续跑。').pack(anchor='w',pady=(8,20))
        self.source=tk.StringVar();self.output=tk.StringVar();self.patients=tk.StringVar()
        self.audit=tk.BooleanVar();self.force=tk.BooleanVar()
        self.inputs=[]
        for title,var,choose in [('输入文件夹',self.source,self.pick_source),('输出文件夹',self.output,self.pick_output)]:
            ttk.Label(frame,text=title).pack(anchor='w')
            row=ttk.Frame(frame);row.pack(fill='x',pady=(4,12))
            entry=ttk.Entry(row,textvariable=var,font=('Microsoft YaHei UI',10));entry.pack(side='left',fill='x',expand=True)
            button=ttk.Button(row,text='浏览…',command=choose);button.pack(side='right',padx=(10,0))
            self.inputs.extend([entry,button])
        options=ttk.Frame(frame);options.pack(fill='x',pady=(0,8))
        for label,var in [('只检查，不转换影像',self.audit),('强制重做（旧结果先备份）',self.force)]:
            b=ttk.Checkbutton(options,text=label,variable=var);b.pack(side='left',padx=(0,20));self.inputs.append(b)
        row=ttk.Frame(frame);row.pack(fill='x',pady=(0,12))
        ttk.Label(row,text='限定患者（可选，留空为全部）').pack(side='left')
        ids=ttk.Entry(row,textvariable=self.patients);ids.pack(side='left',fill='x',expand=True,padx=(10,0));self.inputs.append(ids)
        actions=ttk.Frame(frame);actions.pack(fill='x',pady=(2,16))
        self.start_button=ttk.Button(actions,text='开始 / 续跑',style='Accent.TButton',command=self.start);self.start_button.pack(side='left')
        self.pause_button=ttk.Button(actions,text='暂停',command=self.toggle_pause,state='disabled');self.pause_button.pack(side='left',padx=10)
        self.stop_button=ttk.Button(actions,text='安全停止',command=self.stop,state='disabled');self.stop_button.pack(side='left')
        self.open_button=ttk.Button(actions,text='打开结果',command=self.open_results);self.open_button.pack(side='right')
        self.status=tk.StringVar(value='准备就绪');self.phase=tk.StringVar(value='尚未开始');self.stats=tk.StringVar(value='')
        ttk.Label(frame,textvariable=self.status,font=('Microsoft YaHei UI',12,'bold')).pack(anchor='w')
        self.overall=ttk.Progressbar(frame,maximum=100);self.overall.pack(fill='x',pady=(10,6))
        self.detail=ttk.Progressbar(frame,maximum=100);self.detail.pack(fill='x',pady=(0,6))
        ttk.Label(frame,textvariable=self.phase).pack(anchor='w')
        ttk.Label(frame,textvariable=self.stats).pack(anchor='w',pady=(6,10))
        self.log=tk.Text(frame,height=8,wrap='word',font=('Microsoft YaHei UI',9),background='white',relief='flat',padx=10,pady=8,state='disabled')
        self.log.pack(fill='both',expand=True)
        ttk.Label(frame,text='暂停/停止在安全检查点生效；当前解码或写入可能需要先完成。原始资料始终保留。',font=('Microsoft YaHei UI',9)).pack(anchor='w',pady=(10,0))
        self.settings=PROJECT/'back/interface_settings.json'
        try:
            values=json.loads(self.settings.read_text(encoding='utf-8'))
            self.source.set(values.get('input',''));self.output.set(values.get('output',''));self.patients.set(values.get('patients',''))
        except (OSError,ValueError):pass
        root.protocol('WM_DELETE_WINDOW',self.close);root.after(100,self.poll)
    def pick_source(self):
        value=filedialog.askdirectory(title='选择影像和病例所在的总文件夹',initialdir=self.source.get() or None)
        if value:self.source.set(value)
    def pick_output(self):
        value=filedialog.askdirectory(title='选择空文件夹或之前的结果文件夹；也可在输入框填写新路径',initialdir=self.output.get() or None)
        if value:self.output.set(value)
    def append_log(self,text):
        self.log.configure(state='normal');self.log.insert('end',time.strftime('%H:%M:%S')+'  '+text+'\n')
        if int(self.log.index('end-1c').split('.')[0])>160:self.log.delete('1.0','40.0')
        self.log.see('end');self.log.configure(state='disabled')
    def start(self):
        if self.thread and self.thread.is_alive():return
        source=self.source.get().strip().strip('"');output=self.output.get().strip().strip('"')
        if not source or not output:messagebox.showinfo('请填写路径','请选择输入和输出文件夹。');return
        patients=[x for x in re.split(r'[\s,，;；]+',self.patients.get().strip()) if x]
        args=(source,output,patients or None,self.audit.get(),self.force.get())
        self.control=Control(self.events.put);self.started=time.monotonic();self.last_message=''
        self.overall['value']=0;self.detail.stop();self.detail['value']=0;self.status.set('正在开始…')
        for widget in self.inputs+[self.start_button]:widget.configure(state='disabled')
        self.pause_button.configure(state='normal',text='暂停');self.stop_button.configure(state='normal')
        try:
            self.settings.parent.mkdir(exist_ok=True)
            self.settings.write_text(json.dumps({'input':source,'output':output,'patients':self.patients.get()},ensure_ascii=False),encoding='utf-8')
        except OSError:pass
        self.thread=threading.Thread(target=self.work,args=args,daemon=False);self.thread.start()
    def work(self,source,output,patients,audit,force):
        try:
            code=run(source,output,patients,audit,self.control,force)
            self.events.put({'finished':True,'code':code,'message':'处理完成，请查看问题清单。' if code==2 else '全部处理完成。'})
        except StopRequested:self.events.put({'finished':True,'code':130,'message':'已安全停止。再次点击开始，可从已保存的进度继续。'})
        except BaseException as exc:self.events.put({'finished':True,'code':1,'message':f'任务已停止：{exc}\n修复问题后可使用相同路径续跑。'})
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
        if self.closing and (not self.thread or not self.thread.is_alive()):self.root.destroy();return
        self.root.after(100,self.poll)
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
            c=event['counts'];self.stats.set(f'本次转换 {c.get("images_converted",0)} 项  ·  已跳过 {c.get("images_reused",0)+c.get("images_adopted",0)} 项  ·  病例复用 {c.get("documents_reused",0)} 份')
        if event.get('finished'):
            self.detail.stop();self.status.set({0:'全部处理完成',2:'处理完成，请查看问题清单',130:'已安全停止，可以续跑',1:'任务已停止，请查看下方原因，修复后续跑'}.get(event['code'],message))
            if event['code'] in (0,2):self.overall['value']=100;self.detail['value']=100
            for widget in self.inputs+[self.start_button]:widget.configure(state='normal')
            self.pause_button.configure(state='disabled',text='暂停');self.stop_button.configure(state='disabled')
    def open_results(self):
        path=Path(self.output.get().strip().strip('"'))
        if (path/'查看结果.html').is_file():os.startfile(path/'查看结果.html')
        elif path.is_dir():os.startfile(path)
        else:messagebox.showinfo('尚无结果','先选择输出文件夹并运行任务。')
    def close(self):
        if self.thread and self.thread.is_alive():self.closing=True;self.stop()
        else:self.root.destroy()


def main():
    root=tk.Tk();App(root);root.mainloop()


if __name__=='__main__':main()
