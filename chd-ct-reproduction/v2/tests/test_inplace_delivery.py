"""Same-directory entries, nested maintenance scope and V1 isolation."""
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import os
import json

import pytest
from chd_ct_v2.server import updater,code_sync,network
from test_server_update import local_git
from test_code_sync import source,run as upload_git

HERE=Path(__file__).resolve().parents[1]
V1=HERE.parent if HERE.name=='v2' else HERE.parent/'chd-ct-reproduction'


def test_root_entry_needs_only_the_existing_V1_checkout(tmp_path):
    root=tmp_path/'checkout with spaces 50%/chd-ct-reproduction';root.mkdir(parents=True)
    for name in ['start_server_v2.sh','start_server_v2.py','start_server_v2.cmd']:
        shutil.copy2(V1/name,root/name)
    shutil.copytree(V1/'v2',root/'v2',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache','back'))
    for args in [[],['--engine','stunet'],['--engine','multistage']]:
        result=subprocess.run([sys.executable,'-B',str(root/'start_server_v2.py'),*args,'--help'],
            cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',timeout=20)
        assert result.returncode==0,result.stderr
    assert not (root.parent/'chd-ct-panel-v2').exists()
    runtime=tmp_path/'runtime';results=tmp_path/'results'
    query=subprocess.run([sys.executable,'-B',str(root/'start_server_v2.py'),'--engine','stunet',
        '--profile','generic','--runtime',str(runtime),'--results',str(results),'--query','history'],
        cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',timeout=20)
    assert query.returncode==0,query.stderr
    assert not runtime.exists() and not results.exists()


def test_upload_allows_both_versions_of_the_CHD_project():
    for name in ['start_server_v2.sh','start_server_v2.py','start_server_v2.cmd']:
        assert code_sync.allowed('chd-ct-reproduction/'+name)
    assert code_sync.allowed('chd-ct-reproduction/v2/foundation_engine/05_src/stunet_recipe.py')
    for name in ['start_server.sh','start_server.py','src/chd_ct/server/panel.py','configs/science.json']:
        assert code_sync.allowed('chd-ct-reproduction/'+name)
    assert not code_sync.allowed('AAOCA/code.py')
    for name in ['v2/../src/chd_ct/server/panel.py','v2/04_data/label.nii.gz',
                 'v2/foundation_engine/04_data/manifests/case_inventory.json']:
        assert not code_sync.allowed('chd-ct-reproduction/'+name),name


@pytest.fixture
def nested_repository(tmp_path,monkeypatch):
    monkeypatch.setattr(network,'DEFAULT',dict(schema_version=1,proxy='',git_transport='public-mirrors',ssh_key=''))
    remote=tmp_path/'remote';remote.mkdir()
    local_git(remote,'init','-b','main')
    local_git(remote,'config','user.name','Synthetic')
    local_git(remote,'config','user.email','synthetic@example.invalid')
    project=remote/'chd-ct-reproduction/v2';project.mkdir(parents=True)
    (project/'code.py').write_text('old V2\n')
    for name in ['start_server_v2.sh','start_server_v2.py','start_server_v2.cmd']:
        (project.parent/name).write_text('old launcher\n')
    (project.parent/'start_server.sh').write_text('unchanged V1\n')
    local_git(remote,'add','.')
    local_git(remote,'commit','-m','baseline')
    checkout=tmp_path/'checkout';local_git(tmp_path,'clone',str(remote),str(checkout))
    local_git(checkout,'remote','set-url','origin',updater.UPSTREAM)
    (project/'code.py').write_text('new V2\n')
    (project.parent/'start_server_v2.py').write_text('new launcher\n')
    local_git(remote,'commit','-am','V2 only update')
    actual=updater.git
    def offline(where,*args,**kwargs):
        if args[0]=='ls-remote':raise subprocess.TimeoutExpired('synthetic remote check',1)
        return actual(where,*args,**kwargs)
    monkeypatch.setattr(updater,'git',offline)
    monkeypatch.setattr(updater,'MIRRORS',(str(remote),))
    return checkout,remote


def test_nested_update_archives_root_entry_and_leaves_V1_bytes(nested_repository,tmp_path):
    checkout,remote=nested_repository
    project=checkout/'chd-ct-reproduction/v2'
    before=(project.parent/'start_server.sh').read_bytes()
    repo=updater.inspect_repository(project)
    assert repo['subtree']=='chd-ct-reproduction/v2'
    assert updater.update(repo,tmp_path/'runtime')
    assert (project.parent/'start_server.sh').read_bytes()==before
    assert (project.parent/'start_server_v2.py').read_text()=='new launcher\n'
    archives=list((tmp_path/'runtime').glob('updates/*/code-before-update.tar'))
    with tarfile.open(archives[0]) as archive:
        assert archive.extractfile('chd-ct-reproduction/v2/code.py').read()==b'old V2\n'
        assert archive.extractfile('chd-ct-reproduction/start_server_v2.py').read()==b'old launcher\n'


def test_unified_update_includes_V1_and_backs_it_up(nested_repository,tmp_path):
    checkout,remote=nested_repository
    (remote/'chd-ct-reproduction/start_server.sh').write_text('new project V1\n')
    local_git(remote,'commit','-am','unified project update')
    assert updater.update(updater.inspect_repository(checkout/'chd-ct-reproduction/v2'),tmp_path/'runtime')
    assert (checkout/'chd-ct-reproduction/start_server.sh').read_text()=='new project V1\n'
    archive=next((tmp_path/'runtime').glob('updates/*/code-before-update.tar'))
    with tarfile.open(archive) as saved:
        assert saved.extractfile('chd-ct-reproduction/start_server.sh').read()==b'unchanged V1\n'


def test_unified_update_rejects_another_project(nested_repository,tmp_path):
    checkout,remote=nested_repository
    old=local_git(checkout,'rev-parse','HEAD')
    (remote/'AAOCA').mkdir();(remote/'AAOCA/code.py').write_text('another project\n')
    local_git(remote,'add','.')
    local_git(remote,'commit','-m','other project must be rejected')
    with pytest.raises(ValueError,match='范围|之外|V2'):
        updater.update(updater.inspect_repository(checkout/'chd-ct-reproduction/v2'),tmp_path/'runtime')
    assert local_git(checkout,'rev-parse','HEAD')==old


def test_runtime_cannot_be_inside_V1_code_after_nesting(tmp_path):
    sys.path.insert(0,str(HERE/'foundation_engine/05_src'))
    from v2_host_identity import locations
    with pytest.raises(ValueError,match='代码|运行目录'):
        locations(V1/'unwanted-runtime',tmp_path/'results')
    from chd_ct_v2.panel_v2.identity import locations as six_roots
    with pytest.raises(ValueError,match='代码|运行目录'):
        six_roots(dict(runtime=V1/'unwanted-runtime',results=tmp_path/'six-results',prepared=tmp_path/'prepared'))


def test_root_bash_and_legacy_entry_forward_arguments(tmp_path):
    shell=next((str(p) for p in [Path('D:/programtools/Git/bin/bash.exe'),Path('/bin/bash')] if p.is_file()),None)
    if shell is None:pytest.skip('requires installed Bash')
    env=dict(os.environ,CHD_V2_CONTROL_PYTHON=sys.executable)
    for path in [V1/'start_server_v2.sh',V1.parent/'chd-ct-panel-v2/start_server_v2.sh']:
        result=subprocess.run([shell,str(path),'--help'],cwd=tmp_path,env=env,
            capture_output=True,text=True,encoding='utf-8',timeout=20)
        assert result.returncode==0,result.stderr
        assert 'CHD V2 2.2.2' in result.stdout
    old=subprocess.run([sys.executable,str(V1.parent/'chd-ct-panel-v2/start_server.py'),'--help'],
        cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',timeout=20)
    assert old.returncode==0,old.stderr


def test_manifest_covers_root_entry_tampering(tmp_path):
    root=tmp_path/'chd-ct-reproduction';project=root/'v2';project.mkdir(parents=True)
    (project/'code.py').write_text('synthetic code')
    for name in ['start_server_v2.sh','start_server_v2.py','start_server_v2.cmd','V2_README.md']:
        (root/name).write_text('synthetic entry')
    script=HERE/'scripts/verify_v2.py'
    subprocess.run([sys.executable,str(script),'--root',str(project),'--write'],check=True,capture_output=True)
    (root/'start_server_v2.sh').write_text('changed entry')
    result=subprocess.run([sys.executable,str(script),'--root',str(project)],capture_output=True,text=True)
    assert result.returncode!=0
    assert 'entry' in (result.stdout+result.stderr).lower()


def test_V1_entry_still_loads_the_original_panel(tmp_path):
    result=subprocess.run([sys.executable,str(V1/'start_server.py'),'--help'],cwd=tmp_path,
        capture_output=True,text=True,encoding='utf-8',timeout=20)
    assert result.returncode==0,result.stderr
    assert result.stdout and 'CHD V2' not in result.stdout


def test_standalone_copy_also_protects_V1_source(tmp_path,monkeypatch):
    from chd_ct_v2.panel_v2 import identity
    import v2_host_identity as stu
    project=tmp_path/'chd-ct-reproduction/v2'
    marker=project/'src/chd_ct_v2/panel_v2/identity.py'
    marker.parent.mkdir(parents=True);marker.write_text('synthetic package marker')
    monkeypatch.setattr(identity,'__file__',str(marker))
    monkeypatch.setattr(stu,'__file__',str(project/'foundation_engine/05_src/v2_host_identity.py'))
    with pytest.raises(ValueError,match='代码'):
        identity.locations(dict(runtime=project.parent/'runtime',results=tmp_path/'results',prepared=tmp_path/'prepared'))
    with pytest.raises(ValueError,match='代码'):
        stu.locations(project.parent/'runtime',tmp_path/'stu-results')


def test_real_frozen_module_accepts_its_verified_runtime_release(tmp_path):
    from chd_ct_v2.panel_v2 import identity,snapshot,service
    values={name:str(tmp_path/name) for name in ['runtime','results','prepared']}
    instance=identity.claim(values)
    frozen,digest=snapshot.freeze(HERE,Path(values['runtime']))
    plan=dict(v2_roots=values,panel=identity.PANEL,instance=instance,runtime=values['runtime'],
        directory=str(Path(values['results'])/'synthetic'),v2_id='synthetic',
        frozen_code=str(frozen),code_sha256=digest,frozen_files={},steps=[],v2_stage='environment',
        name='synthetic',sources={},source_contract=None,log='synthetic.log',profile={},request={})
    plan['frozen_plan_sha256']=service.plan_digest(plan)
    record=tmp_path/'plan.json';record.write_text(json.dumps(plan),encoding='utf-8')
    script="import sys,json;sys.path.insert(0,sys.argv[1]);from chd_ct_v2.panel_v2.service import validate;validate(json.load(open(sys.argv[2],encoding='utf-8')));print('frozen worker validation passed')"
    result=subprocess.run([sys.executable,'-B','-c',script,str(frozen/'src'),str(record)],
        cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert result.returncode==0,result.stderr
    assert 'frozen worker validation passed' in result.stdout
    (frozen/'src/chd_ct_v2/cli.py').write_text('# changed frozen source\n',encoding='utf-8')
    changed=subprocess.run([sys.executable,'-B','-c',script,str(frozen/'src'),str(record)],
        cwd=tmp_path,capture_output=True,text=True,encoding='utf-8',timeout=30)
    assert changed.returncode!=0 and '冻结代码' in changed.stderr


def test_unified_upload_publishes_both_versions_and_rejects_another_project(source):
    root,project,remote,runtime,calls=source
    (root/'chd-ct-reproduction').mkdir()
    upload_git(root,'mv','chd-ct-panel-v2','chd-ct-reproduction/v2')
    root_entry=root/'chd-ct-reproduction/start_server_v2.py'
    root_entry.write_text('old root entry\n')
    upload_git(root,'add','.')
    upload_git(root,'commit','-m','nested baseline')
    upload_git(root,'push',str(remote),'codex/ct-panel-v2')
    project=root/'chd-ct-reproduction/v2'
    (project/'code.py').write_text('new nested code\n')
    root_entry.write_text('new root entry\n')
    upload_git(root,'commit','-am','nested code and root entry')
    with network.use(network.load(runtime)):
        preview=code_sync.prepare(project)
        assert preview['files']==['chd-ct-reproduction/start_server_v2.py','chd-ct-reproduction/v2/code.py']
        code_sync.publish(preview,runtime)
        assert upload_git(remote,'rev-parse','codex/ct-panel-v2')==upload_git(root,'rev-parse','HEAD')
        (project.parent/'start_server.sh').write_text('V1 is part of this project\n')
        upload_git(root,'add','.')
        upload_git(root,'commit','-m','unified V1 edit')
        code_sync.publish(code_sync.prepare(project),runtime)
        assert upload_git(remote,'rev-parse','codex/ct-panel-v2')==upload_git(root,'rev-parse','HEAD')
        (root/'AAOCA').mkdir();(root/'AAOCA/code.py').write_text('other project\n')
        upload_git(root,'add','.')
        upload_git(root,'commit','-m','reject other project')
        with pytest.raises(ValueError,match='代码范围'):
            code_sync.prepare(project)
    assert not any('--force' in a for a in calls)
