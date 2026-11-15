import os


def subprocess_environment(proxy=None):
    env = {name: value for name, value in os.environ.items()
           if not name.upper().endswith('_PROXY') and not name.upper().startswith(('GIT_', 'PIP_'))
           and name.upper() != 'SSH_ASKPASS'}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM='1', GIT_TERMINAL_PROMPT='0',
               PIP_CONFIG_FILE=os.devnull, NO_PROXY='' if proxy else '*')
    if proxy:
        env.update(HTTP_PROXY=proxy, HTTPS_PROXY=proxy, ALL_PROXY=proxy)
    return env
