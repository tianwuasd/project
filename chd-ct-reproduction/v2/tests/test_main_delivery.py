"""V2 delivered on main keeps its own launcher and branch-local maintenance."""
import subprocess

import pytest
import test_server_update
import test_code_sync

from chd_ct_v2.server import updater,code_sync,network

repository=test_server_update.repository
source=test_code_sync.source


def test_v2_update_from_main_fetches_main_without_switching(repository,tmp_path):
    repo,remote,old,new,calls=repository
    root=repo['root']
    test_server_update.local_git(root,'branch','-m','main')
    test_server_update.local_git(remote,'branch','-m','main')
    repo=updater.inspect_repository(tmp_path/'checkout/chd-ct-panel-v2')
    assert repo['branch']=='main'
    assert updater.update(repo,tmp_path/'runtime')
    assert test_server_update.local_git(root,'branch','--show-current')=='main'
    assert test_server_update.local_git(root,'rev-parse','HEAD')==new
    fetches=[a for a in calls if a[0]=='fetch']
    assert fetches and all(any(str(v).startswith('refs/heads/main:') for v in a) for a in fetches)


def test_main_code_preview_and_upload_stay_on_main(source):
    root,project,remote,runtime,calls=source
    run=test_code_sync.run
    run(root,'branch','-m','main')
    run(remote,'branch','-m','codex/ct-panel-v2','main')
    test_code_sync.change(root)
    with network.use(network.load(runtime)):
        preview=code_sync.prepare(project)
        assert preview['repo']['branch']=='main'
        code_sync.publish(preview,runtime)
    assert run(remote,'rev-parse','main')==run(root,'rev-parse','HEAD')
    assert not any('codex/ct-panel-v2' in str(v) for a in calls for v in a)
    assert not any('--force' in a for a in calls)


def test_main_support_does_not_enable_arbitrary_branch(source):
    root,project,remote,runtime,calls=source
    test_code_sync.run(root,'branch','-m','unpublished-experiment')
    with pytest.raises(ValueError,match='分支'):
        updater.inspect_repository(project)


def test_branch_change_during_backup_is_rejected_before_apply(repository,tmp_path,monkeypatch):
    repo,remote,old,new,calls=repository
    root=repo['root']
    original=updater.git
    def switch_after_archive(where,*args,**kwargs):
        result=original(where,*args,**kwargs)
        if 'archive' in args:
            test_server_update.local_git(root,'branch','-m','main')
        return result
    monkeypatch.setattr(updater,'git',switch_after_archive)
    with pytest.raises(ValueError,match='备份.*改变|应用.*改变'):
        updater.update(repo,tmp_path/'runtime')
    assert test_server_update.local_git(root,'rev-parse','HEAD')==old


def test_stunet_snapshot_rejects_source_changes_during_copy(tmp_path,monkeypatch):
    import sys
    from pathlib import Path
    from unittest.mock import patch
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'foundation_engine/05_src'))
    import task_runtime as jobs
    code=tmp_path/'source'
    for name in ['05_src','06_configs','04_data/manifests']:
        folder=code/name;folder.mkdir(parents=True)
        (folder/'source.txt').write_text('before')
    (code/'bootstrap_server.sh').write_text('synthetic source')
    monkeypatch.setattr(jobs,'CODE',code)
    target=tmp_path/'job/code'
    copy=jobs.shutil.copytree
    def changed(source,destination,**kwargs):
        result=copy(source,destination,**kwargs)
        if Path(source).name=='05_src':(code/'05_src/source.txt').write_text('updated during snapshot')
        return result
    with patch.object(jobs.shutil,'copytree',changed),pytest.raises(ValueError,match='快照|代码'):
        jobs.snapshot(target)
    assert not (target.parent/'code_manifest.json').exists()
