"""Build a self-contained server-upload ZIP from a positive CODE whitelist."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tomllib
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from update import allowed,validate_manifest


def main():
    candidates=[p for p in ROOT.iterdir() if p.is_file()]
    for directory in ('src','configs','docs','tests','scripts'):
        candidates.extend((ROOT/directory).rglob('*'))
    files={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(candidates) if p.is_file() and allowed(p.relative_to(ROOT).as_posix())}
    version=tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
    manifest=dict(project='chd-ct-panel-v2',version=version,files=files,
                  excluded=['outputs','project.local.json','environments','raw images','model weights'],
                  provenance=dict(vista_commit='47016371e8d56a19630585f49c2cc0299259fa34',
                    vista_version='0.7.2',vista_file='live_resources.py prefix before _rank_setting',
                    reference_full_sha256='296ec34090e020c410b79e3f64c20600ace8d0d8a372eeab945bea334738e12f',
                    adapted_resource_sha256=files['src/chd_stunet/_vendor/vista_live_resources.py'],
                    author_excel_sha256=files['src/chd_stunet/data/author_excel.py']))
    validate_manifest(manifest)
    (ROOT/'code-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    folder=ROOT/'outputs/delivery'
    folder.mkdir(parents=True,exist_ok=True)
    target=folder/('chd-ct-panel-v2-code-'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
    with ZipFile(target,'w',compression=ZIP_DEFLATED) as archive:
        for name in list(files)+['code-manifest.json']:
            archive.write(ROOT/name,arcname='chd-ct-panel-v2/'+name)
    result=dict(zip=str(target),files=len(files)+1,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (folder/'latest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
