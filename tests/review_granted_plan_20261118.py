from copy import deepcopy
from pathlib import Path
from time import perf_counter


def check(args):
    from review_models_20261118 import setup
    engine, root, reports = setup(args)
    from chd_segmentation.acceleration import ResourceSession,RuntimeRequest,SafeBoundary
    from chd_segmentation.acceleration.planning import cpu_plan
    from chd_segmentation.artifacts.store import write_json
    from chd_segmentation.config import load_config
    from chd_segmentation.preprocessing.prepare import load_prepared
    from chd_segmentation.training.session import supply_demand,supply_layout
    config = load_config(root/'course/stu.json')
    config.update(work_root=str(root),run_dir=str(root/'checks/granted-plan'))
    config['resources'].update(workers=0,prefetch=1,cache_bytes=0,wait_seconds=15)
    prepared = load_prepared(root/'course/shared-offline.json')
    from chd_segmentation.acceleration.planning import case_estimates
    prediction = case_estimates(config,prepared,'prediction')
    for row in prediction:
        if row['workspace_bytes'] < row['probability_workspace_bytes']+row['restore_workspace_bytes']:
            raise AssertionError('预测总额度未覆盖真实概率图和原图恢复工作集')
    initial = cpu_plan(config,prepared,1)
    minimum = deepcopy(initial)
    minimum.update(admitted=False,workers=[0],cache_bytes=[0],prefetch=1,cpu_slots=1,
        reader_slots=1,workspace_capacity_bytes=initial['case_estimate_bytes'])
    demand = supply_demand(minimum,1,config['resources'],[1])
    request = RuntimeRequest('assistant2-granted-plan','data-only',root/'resources',config['run_dir'],
        dict(config['resources'],max_wait=15),demand)
    began = perf_counter()
    with ResourceSession(request,layout=supply_layout(minimum)) as session:
        applied = cpu_plan(config,prepared,1,estimates=initial['case_estimates'],allocation=session.allocation)
        if not applied['admitted'] or applied['ram_bytes']>session.allocation.memory_bytes:
            raise AssertionError('真实获准最小额度未形成可执行规划')
        session.replan(supply_demand(applied,1,config['resources'],[1]),
            safe_boundary=SafeBoundary('before_case'),layout=supply_layout(applied))
        record = dict(state='completed',optimizer_updates=0,initial_admitted=initial['admitted'],
            minimum_bytes=demand.memory_bytes,applied_ram_bytes=applied['ram_bytes'],
            actual_capacity=session.allocation.memory_bytes,layout=dict(session.handle.state['layout']),
            generation=session.handle.state['plan_generation'],host=applied['facts'],
            peak_task_rss_bytes=session.handle.state['peak_task_rss'])
        record['prediction_cases'] = prediction
    record.update(seconds=perf_counter()-began,cases=['ct_1001','ct_1012'])
    write_json(reports/'granted-plan.json',record)
