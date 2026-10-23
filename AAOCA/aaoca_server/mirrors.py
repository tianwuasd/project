"""Known public download routes. Order is preference, not a speed guarantee."""
from .state import read, write

OFFICIAL = 'https://github.com/tianwuasd/project.git'
CODE = {
    'gh-proxy': 'https://gh-proxy.com/' + OFFICIAL,
    'ghfast': 'https://ghfast.top/' + OFFICIAL,
    'gitclone': 'https://gitclone.com/github.com/tianwuasd/project.git',
    'official': OFFICIAL,
}
PYPI = {
    'tuna': 'https://pypi.tuna.tsinghua.edu.cn/simple',
    'sjtu': 'https://mirror.sjtu.edu.cn/pypi/web/simple',
    'aliyun': 'https://mirrors.aliyun.com/pypi/simple',
}
DEFAULTS = {'code': list(CODE), 'pypi': list(PYPI)}


def validate(settings):
    for kind, choices in [('code', CODE), ('pypi', PYPI)]:
        order = settings.get(kind)
        if (not isinstance(order, list) or not order or any(not isinstance(x, str) for x in order)
                or len(set(order)) != len(order) or not set(order) <= set(choices)):
            raise ValueError(f'{kind} 镜像顺序需为不重复的已知线路名称')
    return {kind: list(settings[kind]) for kind in DEFAULTS}


def load(runtime):
    return validate(read(runtime / 'mirrors.json', DEFAULTS))


def save(runtime, settings):
    write(runtime / 'mirrors.json', validate(settings))


def known_origin(url):
    accepted = [*CODE.values(), 'https://gh-proxy.org/' + OFFICIAL,
                'git@github.com:tianwuasd/project.git']
    normalize = lambda value: value.rstrip('/').removesuffix('.git')
    return normalize(url) in {normalize(value) for value in accepted}
