"""Planner budget and admission threshold are separate, measured in GiB/MiB."""
import math

DEFAULT_MEMORY_GIB = 20
MEMORY_POLICY = 'zmic44-20g-v1'


def required_free_mib(planner_gib,total_mib,minimum_mib=6000):
    budget,total=float(planner_gib),float(total_mib)
    if not math.isfinite(budget) or budget<2 or not math.isfinite(total) or total<=0:
        raise ValueError('规划显存和实测总显存必须是有效正数')
    # CUDA may expose a few MiB less than the nominal capacity.
    if budget*1024>total+64:
        raise ValueError(f'规划显存 {budget:g} GiB 超过该卡总显存 {total/1024:.2f} GiB，请选择容量匹配的获准GPU')
    # At nominal full-card budgets the former +0.5 GiB rule was unreachable.
    # This is an idle-admission check; the actual-size preflight proves capacity.
    return min(max(float(minimum_mib),(budget+.5)*1024),math.floor(total*.95))
