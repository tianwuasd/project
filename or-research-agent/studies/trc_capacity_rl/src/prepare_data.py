"""Download public LaDe Jilin data and freeze a chronological replay benchmark."""
import argparse
import csv
import hashlib
import json
import math
import urllib.request
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE='https://huggingface.co/datasets/Cainiao-AI/LaDe/resolve/main/'
EXPECTED='12e2cf4664dd5b4475d39dddee8872f5a03b3082f08f0eece7f103baee6c6e73'


def km(lon,lat,lon0,lat0):
    return math.hypot((lon-lon0)*111.32*math.cos(math.radians(lat0)),(lat-lat0)*111.32)


def infer_base(rows):
    pairs=[(float(r['accept_gps_lng']),float(r['accept_gps_lat'])) for r in rows]
    pairs=[p for p in pairs if all(math.isfinite(x) for x in p) and -180<=p[0]<=180 and -90<=p[1]<=90]
    if not pairs: raise ValueError('No valid training GPS for depot construction')
    return float(np.median([p[0] for p in pairs])),float(np.median([p[1] for p in pairs]))


def prepare():
    path=ROOT/'data/raw/delivery_jl.csv'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=EXPECTED: raise ValueError('Upstream dataset changed; review and explicitly update frozen hash')
    rows=list(csv.DictReader(path.open(encoding='utf-8-sig')))
    days=sorted(set(r['ds'] for r in rows));train_days=days[:int(.6*len(days))]
    val_days=days[int(.6*len(days)):int(.8*len(days))];test_days=days[int(.8*len(days)):]
    train=[r for r in rows if r['ds'] in train_days]
    region=Counter(r['region_id'] for r in train).most_common(1)[0][0]
    train=[r for r in train if r['region_id']==region]
    lon0,lat0=infer_base(train)
    groups=defaultdict(list);invalid_destinations=0
    for r in rows:
        if r['region_id']!=region: continue
        hm=r['accept_time'].split()[1].split(':');minute=int(hm[0])*60+int(hm[1])
        if not 7*60<=minute<11*60: continue
        lon,lat=float(r['lng']),float(r['lat'])
        if not all(math.isfinite(x) for x in (lon,lat)) or not (-180<=lon<=180 and -90<=lat<=90):
            invalid_destinations+=1;continue
        # The replay reveals a task at its observed courier-acceptance time.
        order={'order_id':hashlib.sha256(r['order_id'].encode()).hexdigest()[:12],
               'slot':(minute-7*60)//5,'distance_km':km(lon,lat,lon0,lat0),
               'deadline_minutes':30.}
        groups[r['ds']].append(order)
    splits={name:{d:sorted(groups[d],key=lambda r:(r['slot'],r['order_id'])) for d in ds if groups[d]}
            for name,ds in [('train',train_days),('validation',val_days),('test',test_days)]}
    meta={'dataset':'LaDe-D Jilin','url':BASE+'delivery/delivery_jl.csv','sha256':digest,
          'license':'Apache-2.0 according to publisher dataset card','raw_rows':len(rows),'raw_days':len(days),
          'region_chosen_using_training_only':region,'base_derived_using_training_only':[lon0,lat0],
          'dropped_missing_destination_coordinates':invalid_destinations,
          'calendar_year':'not provided in selected CSV; dates retain MMDD format',
          'real_fields':['destination coordinates','courier task acceptance timestamps'],
          'assumed_fields':['single drone depot','aircraft/charging/launch resources','30 minute service target','all costs','ground fallback availability','flight speeds and range'],
          'split_summary':{n:{'days':len(d),'orders':sum(map(len,d.values())),'first':min(d),'last':max(d)} for n,d in splits.items()},
          'limitations':'Courier acceptance is not customer order arrival. Distances are local planar approximations. This is public-data-driven counterfactual simulation, not observed drone operations.'}
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    (ROOT/'data/processed/episodes.json').write_text(json.dumps(splits,indent=2,allow_nan=False),encoding='utf-8')
    (ROOT/'data/manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return meta


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--proxy');args=p.parse_args()
    if args.download:
        op=urllib.request.build_opener(urllib.request.ProxyHandler({'http':args.proxy,'https':args.proxy}) if args.proxy else urllib.request.ProxyHandler())
        (ROOT/'data/raw').mkdir(parents=True,exist_ok=True)
        for remote,local in [('delivery/delivery_jl.csv','delivery_jl.csv'),('README.md','dataset_card.md')]:
            with op.open(BASE+remote,timeout=60) as response: content=response.read()
            (ROOT/'data/raw'/local).write_bytes(content)
    print(json.dumps(prepare(),ensure_ascii=False,indent=2))
