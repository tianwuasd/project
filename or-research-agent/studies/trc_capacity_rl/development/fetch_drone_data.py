"""Fetch official CC BY 4.0 CMU drone measurements for subsequent calibration.

Data remain local. Reading these flights is not a confirmation experiment for
dispatch policies and does not establish joint multi-drone disturbance laws.
"""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=[('parameters.csv',26385034,'7daa897a1af7847b592ccb7a459477ee'),
       ('README.txt',26405786,'af6afbb550286df945c9ecaf4c5f72a4'),
       ('flights.zip',26385070,'cd1d267fd9b025219f9633b95158eea2')]


def main(proxy):
    out=ROOT/'data/raw/drone_flights';out.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':proxy,'https':proxy} if proxy else {}))
    manifest={'dataset':'Data Collected with Package Delivery Quadcopter Drone',
              'doi':'10.1184/R1/12683453.v1','license':'CC BY 4.0',
              'metadata_url':'https://api.figshare.com/v2/articles/12683453',
              'retrieved':'2026-10-04','files':[]}
    for name,file_id,md5 in FILES:
        dest=out/name;url=f'https://ndownloader.figshare.com/files/{file_id}'
        if not dest.exists():
            tmp=dest.with_suffix(dest.suffix+'.partial')
            with opener.open(urllib.request.Request(url,headers={'User-Agent':'or-research/1.0'}),timeout=60) as response,tmp.open('wb') as f:
                while block:=response.read(1024*1024):f.write(block)
            if hashlib.md5(tmp.read_bytes()).hexdigest()!=md5:raise ValueError('Download checksum mismatch')
            tmp.replace(dest)
        content=dest.read_bytes()
        if hashlib.md5(content).hexdigest()!=md5:raise ValueError('Existing file checksum mismatch')
        manifest['files'].append({'name':name,'source_url':url,'bytes':len(content),'source_md5':md5,'sha256':hashlib.sha256(content).hexdigest()})
        print(name,len(content),flush=True)
    with zipfile.ZipFile(out/'flights.zip') as z:
        manifest['archive_members']=len(z.namelist())
        # Inspect names only; never execute or blindly extract archive paths.
        print('Archive members:',len(z.namelist()),z.namelist()[:4],flush=True)
    (ROOT/'data/drone_measurement_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--proxy');main(p.parse_args().proxy)
