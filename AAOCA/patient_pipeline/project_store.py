"""Durable project catalog. Append never deletes unseen sources or good outputs."""
import json
from pathlib import Path
import sqlite3

from common import write_json
from patient_pipeline.project_model import VERSION, now


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=str)


class ProjectStore:
    def __init__(self, root):
        self.root=Path(root).resolve()
        marker=self.root/'project.json'
        if self.root.exists() and any(self.root.iterdir()) and not marker.exists():
            allowed={'.pipeline.lock','processing_manifest.json'}
            if not (self.root/'processing_manifest.json').is_file() and any(p.name not in allowed for p in self.root.iterdir()):
                raise ValueError('输出目录不是已登记项目，也不是旧版结果；请选择新目录或原结果目录。')
        self.root.mkdir(parents=True,exist_ok=True)
        if marker.exists():
            data=json.loads(marker.read_text(encoding='utf-8'))
            if data.get('schema')!=1:raise ValueError('不支持此项目版本')
        else:write_json(marker,{'schema':1,'version':VERSION,'created':now()})
        (self.root/'details').mkdir(exist_ok=True)
        self.db=sqlite3.connect(self.root/'details/project.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS sources(key TEXT PRIMARY KEY,kind TEXT,pid TEXT,
                modality TEXT,study TEXT,series TEXT,sop TEXT,payload TEXT);
            CREATE INDEX IF NOT EXISTS source_pid ON sources(pid);
            CREATE INDEX IF NOT EXISTS source_sop ON sources(sop);
            CREATE TABLE IF NOT EXISTS records(collection TEXT,key TEXT,payload TEXT,
                PRIMARY KEY(collection,key));
            CREATE TABLE IF NOT EXISTS history(seq INTEGER PRIMARY KEY,unit TEXT,time TEXT,payload TEXT);
        ''')
        columns={r[1] for r in self.db.execute('PRAGMA table_info(sources)')}
        if 'unit_key' not in columns:self.db.execute('ALTER TABLE sources ADD COLUMN unit_key TEXT')
        self.db.execute('CREATE INDEX IF NOT EXISTS source_unit ON sources(unit_key)')
        self.db.commit()

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.db.commit();self.db.close()
    def commit(self):self.db.commit()

    def get(self, collection, key, default=None):
        row=self.db.execute('SELECT payload FROM records WHERE collection=? AND key=?',(collection,key)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, collection, key, value, commit=True):
        self.db.execute('INSERT OR REPLACE INTO records VALUES(?,?,?)',(collection,key,encode(value)))
        if commit:self.db.commit()

    def items(self, collection):
        for key,payload in self.db.execute('SELECT key,payload FROM records WHERE collection=? ORDER BY key',(collection,)):
            yield key,json.loads(payload)

    def source(self, key):
        row=self.db.execute('SELECT payload FROM sources WHERE key=?',(key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_source(self, source, commit=True):
        r=source;h=r.get('header',{});tags=h.get('tags',{})
        self.db.execute('INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?,?,?)',(
            r['key'],r['kind'],r.get('pid',''),tags.get('Modality',''),tags.get('StudyInstanceUID',''),
            tags.get('SeriesInstanceUID',''),tags.get('SOPInstanceUID',''),encode(r),r.get('unit_key','')))
        if commit:self.db.commit()

    def sources(self, pid=None, kind=None):
        clauses=[];values=[]
        for col,value in [('pid',pid),('kind',kind)]:
            if value is not None:clauses.append(col+'=?');values.append(value)
        sql='SELECT payload FROM sources'+(' WHERE '+' AND '.join(clauses) if clauses else '')+' ORDER BY key'
        for row in self.db.execute(sql,values):yield json.loads(row[0])

    def unit(self, key):return self.get('units',key)

    def unit_sources(self,key):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM sources WHERE unit_key=? ORDER BY key',(key,))]

    def save_unit(self, key, attempt):
        value={**attempt,'key':key,'updated':now()}
        old=self.unit(key)
        self.db.execute('INSERT INTO history(unit,time,payload) VALUES(?,?,?)',(key,now(),encode(value)))
        if old and old.get('status')=='success' and value.get('status')!='success' and not value.get('invalidate_previous'):
            value={**old,'last_attempt':value}
        self.put('units',key,value)
        return value

    def history(self,key):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM history WHERE unit=? ORDER BY seq',(key,))]

    def path(self, relative):
        p=(self.root/relative).resolve()
        if not p.is_relative_to(self.root) or p==self.root:raise ValueError('输出路径越界')
        return p

    def load_legacy(self, root=None, control=None):
        from patient_pipeline.project_legacy import load_legacy
        return load_legacy(self,Path(root or self.root),control)
