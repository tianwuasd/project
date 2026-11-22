import math


def reserve_margin(required_bytes, budget_bytes, preferred_fraction=.5):
    if required_bytes < 0 or not math.isfinite(preferred_fraction) or preferred_fraction < 0:
        raise ValueError('原始内存需求和额外缓冲比例必须有效且非负')
    required = int(required_bytes)
    requested = math.ceil(required * (1 + preferred_fraction))
    reserved = max(required, min(requested, max(0, int(budget_bytes))))
    return dict(raw_required_bytes=required, memory_bytes=reserved,
                requested_memory_bytes=requested, margin_bytes=reserved-required,
                requested_memory_margin_fraction=preferred_fraction,
                memory_margin_fraction=(reserved-required)/required if required else 0.,
                admitted=required <= budget_bytes)


def raw_capacity(resources, facts, occupied_bytes=0):
    limit = min(memory_ceiling(resources, facts),
                facts['total_bytes']-resources.get('reserve_ram_bytes', 4*2**30))
    return max(0, int(limit)-int(occupied_bytes))


def host_fraction(resources):
    maximum = 1. if resources.get('technical_cpu', False) and resources.get('device') == 'cpu' else .8
    return min(maximum, resources.get('ram_fraction', .8))


def memory_ceiling(resources, facts):
    host_limit = int(facts['total_bytes']*host_fraction(resources))
    if resources.get('memory_policy') == 'host_fraction':
        return host_limit
    return min(host_limit, 100_000_000_000, resources.get('memory_limit_bytes', 100_000_000_000))


def project_usage(records, resources):
    host_fraction = resources.get('memory_policy') == 'host_fraction'
    return sum(max(row.get('ram_bytes', 0), row.get('rss_bytes', 0)) for row in records
               if host_fraction or row.get('memory_policy') != 'host_fraction')
