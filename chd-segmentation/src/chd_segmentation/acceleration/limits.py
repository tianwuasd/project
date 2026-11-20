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
