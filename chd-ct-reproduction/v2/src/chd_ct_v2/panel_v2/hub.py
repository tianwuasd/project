"""Step-first CHD V2 panel; backends retain their separate task identities."""
import json
import os
from pathlib import Path
import sys

from ..server.prompts import ask,PanelBack,UserExit

PROJECT=Path(__file__).resolve().parents[3]
STAGES={'1':'prepare','2':'train','3':'predict','4':'predict-all','5':'diagnosis','6':'evaluate','a':'assessment','e':'environment'}
QUERIES={'s':'status','g':'gpu','m':'models','l':'logs','p':'processes','o':'results','h':'history'}
ACTIONS={'r':'resume','k':'rerun','t':'stop','d':'trash','z':'diagnostics','v':'upload','c':'network','u':'update','w':'push-code'}


def choose_method(default='stunet'):
    print('本步骤使用的方法：1 STU-Net-S  2 原六路  b 返回')
    value=ask('选择 ['+('1' if default=='stunet' else '2')+']：')
    value=value or ('1' if default=='stunet' else '2')
    if value not in {'1','2'}:raise ValueError('请选择1或2')
    return 'stunet' if value=='1' else 'multistage'


def dispatch_stage(stage,method):
    if method=='stunet':
        if stage in {'predict-all','diagnosis','evaluate'}:
            raise ValueError('STU-Net当前提供七结构外折预测/评估，请选择3；病型诊断沿用原六路任务。')
        from .integrated import run_stunet
        args=['--server-defaults','--assessment'] if stage=='assessment' else ['--server-defaults','--stage',stage]
        return run_stunet(args,unified=True)
    from . import panel
    profile=json.loads(Path(panel.SERVER_PROFILE).read_text(encoding='utf-8'))
    values=dict(runtime=profile['runtime'],results=profile['results'],prepared=profile['prepared'])
    panel.check(values)
    args=panel.parser().parse_args(['--stage',stage])
    return panel.new_task(values,args,profile,interactive=True)


def dispatch_operation(key,method):
    if method=='stunet':
        from .integrated import run_stunet
        args=['--server-defaults']
        if key in QUERIES:args+=['--query',QUERIES[key]]
        elif key in {'u','w','c'}:args+=[{'u':'--update','w':'--push-code','c':'--configure-network'}[key]]
        else:
            sys.path.insert(0,str(PROJECT/'foundation_engine/05_src'))
            import task_panel,server_profile
            profile=server_profile.load()
            previous=os.environ.get('VISTA_CATALOG_VERSION')
            os.environ['VISTA_CATALOG_VERSION']='2'
            try:target=task_panel.choose_job(profile['runtime'])
            except task_panel.Back:raise PanelBack from None
            except task_panel.Quit:raise UserExit from None
            finally:
                if previous is None:os.environ.pop('VISTA_CATALOG_VERSION',None)
                else:os.environ['VISTA_CATALOG_VERSION']=previous
            if key=='z':
                from diagnostic_export import export
                print('诊断包已保存：'+str(export(target,Path(profile['runtime'])/'diagnostic-exports')))
                return 0
            if key=='d':
                from pipeline_catalog import trash
                print('将任务移入可恢复回收区，暂不释放磁盘空间：'+target)
                if ask('输入 DELETE 确认回收；其它输入返回：')=='DELETE':
                    print('已移入回收区：'+str(trash(profile['runtime'],target)))
                return 0
            args+=[{'r':'--resume-job','k':'--rerun-job','t':'--stop','v':'--upload-job'}[key],target] if key in {'r','k','v'} else ['--stop','--job',target]
        return run_stunet(args,unified=True)
    from . import panel
    profile=json.loads(Path(panel.SERVER_PROFILE).read_text(encoding='utf-8'))
    values=dict(runtime=profile['runtime'],results=profile['results'],prepared=profile['prepared'])
    args=panel.parser().parse_args([])
    if key in QUERIES:
        args.query=QUERIES[key]
        if key not in {'h','g','p'}:args.job=panel.select(values)
        return panel.query(values,args,profile)
    args.action=ACTIONS[key]
    if key not in {'u','w','c'}:args.job=panel.select(values)
    return panel.action(values,args,profile)


def run():
    from .integrated import VERSION
    preferred='stunet'
    while True:
        print(f'\nCHD V2 {VERSION} | 统一流程面板')
        print('1 预处理  2 训练  3 预测/评估  4 全病例预测  5 诊断  6 重算评估  a 完整测评  e 环境')
        print('s 进度  g 显卡  m 模型  l 日志  p 进程  o 结果  h 历史')
        print('r 续跑  k 重跑  t 停止  d 回收  z 本地诊断  v 上传产物  c 网络  u 更新项目  w 上传代码  q/0 退出')
        print('路径按服务器配置自动使用；新任务在所选方法自己的环境与结果根内保存。')
        try:
            key=ask('请选择 [s]：').lower() or 's'
            if key in {'0','q'}:return 0
            if key in STAGES:
                method=choose_method(preferred);preferred=method
                dispatch_stage(STAGES[key],method)
            elif key in QUERIES or key in {'r','k','t','d','z','v'}:
                method=choose_method(preferred);preferred=method
                dispatch_operation(key,method)
            elif key in {'u','w','c'}:
                result=dispatch_operation(key,preferred)
                if key=='u' and result==0:
                    print('请重新运行 start_server_v2.sh 载入更新后的完整代码。')
                    return 0
            else:print('请输入菜单中的选项')
        except PanelBack:continue
        except (UserExit,EOFError):return 0
        except KeyboardInterrupt:print('已返回统一面板，后台任务继续。')
        except (ValueError,OSError,RuntimeError,KeyError) as error:print('未执行/未完成：'+str(error))
