"""One step-first panel, configured paths and method-specific dispatch."""
from pathlib import Path
from unittest.mock import patch
import pytest


def test_home_selects_step_before_method_and_does_not_open_two_panels(capsys):
    from chd_ct_v2.panel_v2 import integrated,hub
    with patch('sys.stdin.isatty',return_value=True),patch('builtins.input',side_effect=['1','1','q']),\
         patch.object(hub,'dispatch_stage',return_value=0) as submit:
        assert integrated.main([])==0
    assert submit.call_args.args[:2]==('prepare','stunet')
    text=capsys.readouterr().out
    assert text.index('1 预处理')<text.index('本步骤使用的方法')


def test_each_stage_can_choose_the_other_method_without_switching_panels():
    from chd_ct_v2.panel_v2 import integrated,hub
    with patch('sys.stdin.isatty',return_value=True),patch('builtins.input',side_effect=['2','2','q']),\
         patch.object(hub,'dispatch_stage',return_value=0) as submit:
        assert integrated.main([])==0
    assert submit.call_args.args[:2]==('train','multistage')


def test_configured_prepare_does_not_ask_for_paths_or_create_descriptor(tmp_path):
    import sys
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'foundation_engine/05_src'))
    import task_panel
    raw=tmp_path/'raw';raw.mkdir();descriptor=tmp_path/'inputs'
    profile=dict(name='zmic44',runtime=str(tmp_path/'runtime'),results=str(tmp_path/'results'),
        data=str(descriptor),raw_data=str(raw),conda='',gpu_memory=20)
    args=task_panel.make_parser().parse_args([])
    with patch('server_profile.load',return_value=profile),\
         patch.object(task_panel,'ask',side_effect=AssertionError('configured paths must not be prompted')):
        config=task_panel.stage_config(args,{},'prepare',interactive=True)
    assert config['data']==str(descriptor)
    assert config['input_bootstrap']['raw']==str(raw)
    assert config['input_bootstrap']['seed']==42
    assert not descriptor.exists()


def test_server_defaults_point_to_confirmed_raw_source():
    import sys,json
    root=Path(__file__).resolve().parents[1]
    profile=json.loads((root/'foundation_engine/06_configs/servers/zmic44.json').read_text(encoding='utf-8'))
    assert profile['raw_data']=='/data_nas/zhangruichen/baidu_import/ImageCHD_dataset'
    assert profile['data']=='/data_nas/zhangruichen/chd_ct_v2_stunet_inputs'
    assert profile['pretrained_weights'].endswith('/model_weights/stunet-small/small_ep4k.model')


def test_standard_author_Excel_name_is_discovered(tmp_path):
    from chd_ct_v2.imagechd.spacing import find_spacing_excel
    image=tmp_path/'ct_1001_image.nii.gz';image.write_bytes(b'header not needed')
    excel=tmp_path/'imagechd_dataset_image_info.xlsx';excel.write_bytes(b'synthetic')
    assert find_spacing_excel(image)==excel


@pytest.mark.parametrize('stage',['environment','prepare','train','predict'])
def test_stunet_dispatches_selected_stage_without_opening_a_backend_menu(stage):
    from chd_ct_v2.panel_v2 import integrated,hub
    with patch.object(integrated,'run_stunet',return_value=0) as run:
        assert hub.dispatch_stage(stage,'stunet')==0
    assert run.call_args.args[0]==['--server-defaults','--stage',stage]


def test_six_path_dispatches_stage_directly_with_its_configured_roots():
    from chd_ct_v2.panel_v2 import panel,hub
    with patch.object(panel,'new_task',return_value=0) as run,patch.object(panel,'check'):
        assert hub.dispatch_stage('prepare','multistage')==0
    values,args,profile=run.call_args.args
    assert args.stage=='prepare'
    assert values['runtime']==profile['runtime']
    assert run.call_args.kwargs['interactive']


def test_environment_defaults_are_noninteractive_and_no_descriptor_is_created(tmp_path):
    import task_panel
    profile=dict(name='zmic44',runtime=str(tmp_path/'runtime'),results=str(tmp_path/'results'),
        data=str(tmp_path/'inputs'),raw_data=str(tmp_path/'raw'),conda='',gpu_memory=20)
    args=task_panel.make_parser().parse_args([])
    with patch('server_profile.load',return_value=profile),patch.object(task_panel,'ask',side_effect=AssertionError('path prompt')):
        config=task_panel.stage_config(args,{},'environment',interactive=True)
    assert config['tasks']==['environment']
    assert config['catalog_schema']==2 and config['stage']=='environment'
    assert not (tmp_path/'inputs').exists()


@pytest.mark.parametrize('signal,expected',[('Quit','UserExit'),('Back','PanelBack')])
def test_stunet_form_navigation_propagates_to_unified_home(signal,expected):
    import sys
    from chd_ct_v2.panel_v2 import integrated,hub
    from chd_ct_v2.server import prompts
    sys.path.insert(0,str(integrated.PROJECT/'foundation_engine/05_src'))
    import task_panel
    with patch.object(task_panel,'main',side_effect=getattr(task_panel,signal)),\
         pytest.raises(getattr(prompts,expected)):
        hub.dispatch_stage('prepare','stunet')


@pytest.mark.parametrize('signal,expected',[('Quit','UserExit'),('Back','PanelBack')])
def test_stunet_task_selection_obeys_shared_navigation(signal,expected):
    import sys
    from chd_ct_v2.panel_v2 import integrated,hub
    from chd_ct_v2.server import prompts
    sys.path.insert(0,str(integrated.PROJECT/'foundation_engine/05_src'))
    import task_panel
    with patch.object(task_panel,'choose_job',side_effect=getattr(task_panel,signal)),\
         pytest.raises(getattr(prompts,expected)):
        hub.dispatch_operation('r','stunet')


def test_update_exits_old_control_process_before_another_submission():
    from chd_ct_v2.panel_v2 import hub
    with patch('builtins.input',side_effect=['u']),patch.object(hub,'dispatch_operation',return_value=0) as update:
        assert hub.run()==0
    assert update.call_args.args[0]=='u'


def test_pretrained_default_follows_effective_runtime_without_touching_default_instance(tmp_path):
    import sys
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'foundation_engine/05_src'))
    import task_panel
    raw=tmp_path/'raw';raw.mkdir()
    default=tmp_path/'default-runtime';custom=tmp_path/'custom-runtime'
    profile=dict(name='zmic44',runtime=str(default),results=str(tmp_path/'results'),
        data=str(tmp_path/'inputs'),raw_data=str(raw),conda='',gpu_memory=20,
        pretrained_weights=str(default/'model_weights/stunet-small/small_ep4k.model'))
    args=task_panel.make_parser().parse_args(['--runtime',str(custom),'--results',str(tmp_path/'custom-results')])
    with patch('server_profile.load',return_value=profile):
        config=task_panel.stage_config(args,{},'assessment',interactive=False)
    assert config['runtime']==str(custom)
    assert config['pretrained_weights']==str(custom/'model_weights/stunet-small/small_ep4k.model')
    assert not default.exists()


def test_recycle_cancel_does_not_mutate_jobs(tmp_path):
    from chd_ct_v2.panel_v2 import hub
    with patch('task_panel.choose_job',return_value=str(tmp_path/'job')),\
         patch('pipeline_catalog.trash',side_effect=AssertionError('cancelled')) as move,\
         patch('builtins.input',return_value=''):
        assert hub.dispatch_operation('d','stunet')==0
    move.assert_not_called()


def test_diagnostic_action_delegates_to_existing_local_exporter(tmp_path):
    from chd_ct_v2.panel_v2 import hub
    with patch('task_panel.choose_job',return_value=str(tmp_path/'job')),\
         patch('diagnostic_export.export',return_value=tmp_path/'diagnostic.zip') as export:
        assert hub.dispatch_operation('z','stunet')==0
    assert export.call_args.args[0]==str(tmp_path/'job')
