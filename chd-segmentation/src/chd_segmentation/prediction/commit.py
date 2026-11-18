import os
from pathlib import Path

from ..artifacts.store import read_json,write_json


class CaseCommit:
    def __init__(self,root,task):
        self.root,self.task = Path(root).resolve(),task
        self.journal = self.root/'pending-case.json'

    def path(self,name):
        if not isinstance(name,str) or Path(name).name!=name:
            raise ValueError('病例提交路径必须是结果目录内的文件名')
        path = self.root/name
        if path.is_symlink() or path.resolve().parent!=self.root:
            raise ValueError('病例提交路径不属于当前结果目录')
        return path

    def begin(self,case_id,receipt,final,pending):
        if self.journal.exists() or self.path(final).exists() or self.path(pending).exists():
            raise FileExistsError('病例提交目标已存在，需先恢复当前提交')
        write_json(self.journal,dict(task=self.task,case_id=case_id,receipt=receipt,
            final=final,pending=pending,phase='writing'))

    def ready(self,receipt):
        record = read_json(self.journal)
        if record['task']!=self.task or not self.path(record['pending']).is_file():
            raise ValueError('病例提交身份或已写文件不完整')
        write_json(self.journal,dict(record,receipt=receipt,phase='ready'))

    def recover(self,completed,save):
        if not self.journal.exists():
            return
        record = read_json(self.journal)
        if record['task']!=self.task or record['phase'] not in ('writing','ready'):
            raise ValueError('待提交病例属于其他任务或状态无效')
        final,pending = self.path(record['final']),self.path(record['pending'])
        if record['phase']=='writing':
            if final.exists():
                raise FileExistsError('未完成写入对应的最终文件已存在')
            if pending.exists():
                pending.unlink()
        else:
            if not final.exists():
                if not pending.is_file():
                    raise FileNotFoundError('已完成写入的病例文件缺失')
                os.replace(pending,final)
            elif pending.exists():
                raise FileExistsError('同一病例同时存在待提交和最终文件')
            name = record['case_id']
            if name in completed and completed[name]!=record['receipt']:
                raise ValueError('待提交病例与已有完成记录不同')
            completed[name] = record['receipt']
            save(completed)
        self.journal.unlink()
