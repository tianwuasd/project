"""New V2 entry is independent of V1 and existing frozen CT tasks."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
from contextlib import contextmanager

import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_new_shell_resolves_from_its_own_path():
    shell=(ROOT/'start_server_v2.sh').read_text(encoding='utf-8')
    assert 'BASH_SOURCE' in shell and 'start_server_v2.py' in shell
    assert 'chd-ct-reproduction' not in shell


def test_cli_dispatches_explicit_methods_without_installing():
    from chd_ct_v2.panel_v2 import integrated
    with patch.object(integrated,'run_stunet',return_value=0) as stu:
        assert integrated.main(['--engine','stunet','--query','history'])==0
        stu.assert_called_once_with(['--query','history'])
    with patch.object(integrated,'run_multistage',return_value=0) as ct:
        assert integrated.main(['--engine','multistage','--query','history'])==0
        ct.assert_called_once_with(['--query','history'])


def test_new_help_requires_only_stdlib():
    result=subprocess.run([sys.executable,'-B',str(ROOT/'start_server_v2.py'),'--help'],
        cwd=ROOT.parent,capture_output=True,text=True,encoding='utf-8')
    assert result.returncode==0,result.stderr
    assert 'stunet' in result.stdout and 'multistage' in result.stdout


def test_stunet_menu_quit_returns_cleanly_without_creating_state(tmp_path):
    from chd_ct_v2.panel_v2 import integrated
    runtime,results=tmp_path/'runtime',tmp_path/'results'
    with patch('sys.stdin.isatty',return_value=True),patch('builtins.input',return_value='q'):
        assert integrated.main(['--engine','stunet','--profile','generic','--runtime',str(runtime),
            '--results',str(results)])==0
    assert not runtime.exists() and not results.exists()


def test_engine_namespace_rejects_v1_and_legacy_v2_before_write(tmp_path):
    sys.path.insert(0,str(ROOT/'foundation_engine/05_src'))
    host=importlib.import_module('v2_host_identity')
    for marker in ['last-job.json','.ct-panel-v2.json','settings_v2.json']:
        runtime=tmp_path/marker.replace('.','_');runtime.mkdir()
        (runtime/marker).write_text('{}',encoding='utf-8')
        results=tmp_path/(runtime.name+'_results')
        before={p.name:p.read_bytes() for p in runtime.iterdir()}
        with pytest.raises(ValueError):host.claim(runtime,results)
        assert not results.exists()
        assert before=={p.name:p.read_bytes() for p in runtime.iterdir()}


def test_engine_namespace_readonly_and_explicit_claim(tmp_path):
    sys.path.insert(0,str(ROOT/'foundation_engine/05_src'))
    host=importlib.import_module('v2_host_identity')
    runtime,results=tmp_path/'runtime',tmp_path/'results'
    assert host.check(runtime,results) is None
    assert not runtime.exists() and not results.exists()
    identity=host.claim(runtime,results)
    assert identity and host.check(runtime,results)==identity
    assert host.claim(runtime,results)==identity
    with pytest.raises(ValueError):host.claim(runtime,tmp_path/'other_results')
    assert not (tmp_path/'other_results').exists()
    with pytest.raises(ValueError):host.claim(runtime,runtime/'nested')
    assert host.bind_runtime(runtime,None,tmp_path/'wrong_default')==str(results.resolve())


def test_namespace_rejects_corrupt_and_redirected_marker(tmp_path):
    sys.path.insert(0,str(ROOT/'foundation_engine/05_src'))
    host=importlib.import_module('v2_host_identity')
    runtime,results=tmp_path/'runtime',tmp_path/'results';runtime.mkdir()
    (runtime/host.MARKER).write_text('[]',encoding='utf-8')
    with pytest.raises(ValueError):host.check(runtime,results)
    assert not results.exists()


def test_nested_runtime_data_is_rejected_without_creation(tmp_path):
    sys.path.insert(0,str(ROOT/'foundation_engine/05_src'))
    host=importlib.import_module('v2_host_identity')
    with pytest.raises(ValueError):host.claim(tmp_path/'data/runtime',tmp_path/'results',tmp_path/'data')
    assert not (tmp_path/'data').exists()


def test_embedded_code_upload_refuses_private_nested_manifests():
    from chd_ct_v2.server.code_sync import allowed
    assert allowed('chd-ct-panel-v2/start_server_v2.cmd')
    assert allowed('chd-ct-panel-v2/foundation_engine/tests/test_private_update.py')
    assert allowed('chd-ct-panel-v2/foundation_engine/05_src/stunet_trainers.py')
    assert not allowed('chd-ct-panel-v2/foundation_engine/04_data/manifests/case_inventory.json')
    assert not allowed('chd-ct-panel-v2/foundation_engine/09_reports/metrics.json')


def test_engine_maintenance_scope_is_only_v2():
    sys.path.insert(0,str(ROOT/'foundation_engine/05_src'))
    import v2_host_maintenance as maintenance
    assert maintenance.BRANCH=='main'
    assert maintenance.SUBDIR=='chd-ct-panel-v2'
    import task_update
    with patch.object(maintenance,'update',return_value=0) as update:
        task_update.update('/synthetic/runtime')
        update.assert_called_once_with('/synthetic/runtime')


def test_submit_freezes_integrated_code_and_own_registry(tmp_path):
    sys.path.insert(0,str(ROOT/'foundation_engine/05_src'))
    import task_panel
    import task_runtime as jobs
    import pipeline_catalog
    runtime,results,data=tmp_path/'runtime',tmp_path/'results',tmp_path/'input'
    data.mkdir()
    args=task_panel.make_parser().parse_args(['--stage','prepare','--data-dir',str(data),
        '--runtime',str(runtime),'--results',str(results),'--models','S1','--profile','generic',
        '--non-interactive'])
    config=task_panel.stage_config(args,{},'prepare',interactive=False)
    with patch.object(jobs,'launch',return_value=0):
        assert jobs.submit(config)==0
    rows=pipeline_catalog.history(runtime)
    assert len(rows)==1
    plan=jobs.load_plan(rows[0]['directory'])
    assert plan['methods']==['S1'] and plan['global_batch_size']==4
    assert (Path(plan['code'])/'05_src/v2_host_identity.py').is_file()
    assert (runtime/'pipeline_v2/index.json').is_file()
    assert not (runtime/'last-job.json').exists()
    assert plan['code']!=str(ROOT)


def test_two_method_query_roots_stay_readonly(tmp_path):
    for engine in ('stunet','multistage'):
        runtime,results=tmp_path/(engine+'_runtime'),tmp_path/(engine+'_results')
        arguments=[sys.executable,'-B',str(ROOT/'start_server_v2.py'),'--engine',engine,
            '--query','history','--runtime',str(runtime),'--results',str(results)]
        if engine=='stunet':arguments+=['--profile','generic']
        else:arguments+=['--prepared-root',str(tmp_path/'ct_prepared')]
        completed=subprocess.run(arguments,capture_output=True,text=True,encoding='utf-8')
        assert completed.returncode==0,completed.stderr
        assert not runtime.exists() and not results.exists()


def test_first_network_configuration_retains_owned_runtime_on_reopen(tmp_path):
    from chd_ct_v2.panel_v2 import integrated
    runtime,results=tmp_path/'runtime',tmp_path/'results'
    common=['--engine','stunet','--profile','generic','--runtime',str(runtime),
        '--results',str(results)]
    assert integrated.main(common+['--configure-network','--github-transport','github-https'])==0
    assert integrated.main(common+['--query','history'])==0
    assert (runtime/'.ct-v2-stunet.json').is_file()
    assert (results/'.ct-v2-stunet.json').is_file()


def test_stage_submissions_keep_the_instance_result_root(tmp_path):
    from chd_ct_v2.panel_v2 import integrated
    sys.path.insert(0,str(ROOT/'foundation_engine/05_src'))
    import task_panel
    import task_runtime as jobs
    import v2_host_identity as host
    import pipeline_catalog
    runtime,results,data=tmp_path/'runtime',tmp_path/'results',tmp_path/'input'
    data.mkdir()
    common=['--engine','stunet','--profile','generic','--runtime',str(runtime),'--results',str(results)]
    assert integrated.main(common+['--configure-network','--github-transport','github-https'])==0
    args=task_panel.make_parser().parse_args(common[2:]+['--data-dir',str(data)])
    for _ in range(2):
        config=task_panel.stage_config(args,{},'prepare',interactive=False)
        with patch.object(jobs,'launch',return_value=0):
            assert jobs.submit(config)==0
        assert host.bind_runtime(runtime,None,tmp_path/'unused')==str(results.resolve())
        assert integrated.main(common+['--query','history'])==0
    assert len(pipeline_catalog.history(runtime))==2
    assert not (results/'pipeline_v2/prepare/pipeline_v2').exists()
    redirected=dict(config,results=str(tmp_path/'foreign-output'))
    with pytest.raises(ValueError):jobs.validate_request(redirected)
    assert not (tmp_path/'foreign-output').exists()


def test_environment_job_is_visible_in_the_same_catalog_and_lock(tmp_path):
    sys.path.insert(0,str(ROOT/'foundation_engine/05_src'))
    import task_panel
    import task_runtime as jobs
    import pipeline_catalog
    runtime,results,data=tmp_path/'runtime',tmp_path/'results',tmp_path/'input'
    data.mkdir()
    args=task_panel.make_parser().parse_args(['--profile','generic','--runtime',str(runtime),
        '--results',str(results),'--data-dir',str(data),'--tasks','environment'])
    config=task_panel.new_config(args,{},interactive=False)
    with patch.object(jobs,'launch',return_value=0) as launch:
        assert jobs.submit(config)==0
        assert Path(launch.call_args.args[1].name).parent==runtime/'pipeline_v2'
    rows=pipeline_catalog.history(runtime)
    assert len(rows)==1 and rows[0]['stage']=='environment'
    assert not (runtime/'last-job.json').exists()


def test_stunet_maintenance_uses_its_actual_task_lock(tmp_path):
    sys.path.insert(0,str(ROOT/'foundation_engine/05_src'))
    import v2_host_identity as host
    import v2_host_maintenance as maintenance
    import task_runtime as jobs
    from chd_ct_v2.server import updater,code_sync
    runtime,results=tmp_path/'runtime',tmp_path/'results'
    host.claim(runtime,results)
    entered=[]
    @contextmanager
    def guard(path):
        entered.append(Path(path))
        raise ValueError('synthetic held STU task lock')
        yield
    with patch.object(jobs,'runtime_lock',guard),patch.object(updater,'inspect_repository') as inspect:
        with pytest.raises(ValueError,match='synthetic held STU task lock'):maintenance.update(runtime)
        inspect.assert_not_called()
    args=type('Args',(),dict(runtime=str(runtime),results=str(results)))()
    with patch.object(jobs,'runtime_lock',guard),patch.object(code_sync,'run') as run:
        with pytest.raises(ValueError,match='synthetic held STU task lock'):
            maintenance.push_code(args,dict(runtime=str(runtime),results=str(results)),False)
        run.assert_not_called()
    assert entered==[runtime/'pipeline_v2']*2


def test_update_keeps_the_explicit_results_root(tmp_path):
    from chd_ct_v2.panel_v2 import integrated
    sys.path.insert(0,str(ROOT/'foundation_engine/05_src'))
    import v2_host_identity as host
    from chd_ct_v2.server import updater
    runtime,results=tmp_path/'runtime',tmp_path/'results'
    with patch.object(host,'claim') as claim,patch.object(updater,'inspect_repository',return_value={}),\
            patch.object(updater,'update',return_value=0):
        assert integrated.main(['--engine','stunet','--profile','generic','--runtime',str(runtime),
            '--results',str(results),'--update'])==0
        claim.assert_called_once_with(str(runtime),str(results))


def test_entry_does_not_normalize_away_unsafe_runtime_components(tmp_path):
    from chd_ct_v2.panel_v2 import integrated
    runtime=tmp_path/'alias/../runtime'
    assert integrated.main(['--engine','stunet','--profile','generic','--runtime',str(runtime),
        '--results',str(tmp_path/'results'),'--query','history'])==1
    assert not (tmp_path/'runtime').exists()
