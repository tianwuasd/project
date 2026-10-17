"""Explicit, local identity rules authorized by the dataset owner."""
from functools import lru_cache
import json
import os
import re
import unicodedata
from pathlib import Path


def name_variants(value):
    raw=str(value or '').strip()
    candidates=[raw]
    try:
        decoded=raw.encode('latin1').decode('gb18030')
        if decoded.encode('gb18030').decode('latin1')==raw:candidates.append(decoded)
    except (UnicodeEncodeError,UnicodeDecodeError):pass
    return {re.split(r'[/=^]',unicodedata.normalize('NFKC',s),1)[0].strip() for s in candidates if s.strip()}


@lru_cache(maxsize=1)
def load_policy():
    path=os.environ.get('AAOCA_REPAIR_POLICY')
    return json.loads(Path(path).read_text(encoding='utf-8')) if path else None


def apply_roster(roster,issues):
    policy=load_policy()
    if not policy:return roster,issues
    for ident,change in policy['roster_overrides'].items():
        if ident not in roster:raise ValueError(f'Approved roster ID missing: {ident}')
        r=roster[ident]
        r.update(name=change['name'],name_aliases=change['name_aliases'],
                 aliases=sorted(set(r['aliases'])|set(change['aliases'])),
                 fields=change['fields'],conflict=False,
                 confirmed_source_records=change['source_records'],
                 field_history=change['field_history'],identity_resolution='user_confirmed_same_patient')
    for old,new in policy['id_aliases'].items():
        if new not in roster:raise ValueError(f'Canonical roster ID missing: {new}')
        roster[new]['aliases']=sorted(set(roster[new].get('aliases',[]))|{old})
    for r in roster.values():
        r['_name_merge_approved']=True
        r['name_aliases']=sorted(set(r.get('name_aliases',[]))|{r.get('name','')})
        r['repair_policy_version']=policy['version']
    accepted=set(policy['roster_overrides'])
    out=[]
    for issue in issues:
        if issue.get('code')=='ROSTER_ID_CONFLICT' and issue.get('patient_id') in accepted:
            out.append({**issue,'code':'ROSTER_ID_MERGED_BY_USER','severity':'info',
                        'message':'用户确认同一编号内的记录为同一患者，已合并并保留全部姓名和住院号别名。'})
        else:out.append(issue)
    return roster,out


def approved_name_patient(name,roster):
    hits={ident for ident,r in roster.items() if r.get('_name_merge_approved')
          and name and name in r.get('name_aliases',[r.get('name','')])}
    return next(iter(hits)) if len(hits)==1 else None
