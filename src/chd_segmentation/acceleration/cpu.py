from .contracts import CaseExecution


def choose_execution(estimate, snapshot, policy):
    available = max(0, snapshot.headroom_bytes - estimate.fixed_bytes)
    if policy.execution not in ('auto', 'whole', 'blocks'):
        raise ValueError('预处理execution必须为auto、whole或blocks')
    if snapshot.cancelled:
        return CaseExecution('wait', 'cancelled', 0, 0, 0)
    if estimate.whole_peak_bytes <= available and policy.execution != 'blocks':
        return CaseExecution('whole', '整例工作集在当前额度内', estimate.whole_peak_bytes, 0, 1)
    if policy.execution == 'whole' or available < estimate.minimum_block_bytes:
        return CaseExecution('wait', '等待整例或最小合法块额度', 0, 0, 0)
    threads = max(1, policy.max_threads)
    block = available // threads
    if policy.block_bytes is not None:
        block = min(block, policy.block_bytes)
    if policy.max_block_bytes is not None:
        block = min(block, policy.max_block_bytes)
    if block < estimate.minimum_block_bytes:
        if any(value is not None and value < estimate.minimum_block_bytes for value in (policy.block_bytes, policy.max_block_bytes)):
            return CaseExecution('wait', '显式块预算小于当前算法最小合法块', 0, 0, 0)
        threads = max(1, available // estimate.minimum_block_bytes)
        block = available // threads
    return CaseExecution('blocks', '整例工作集超过当前额度', available, block, threads)
