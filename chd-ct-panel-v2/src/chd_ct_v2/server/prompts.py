"""Navigation shared by the panel and its nested input forms."""


class PanelBack(Exception):
    pass


class UserExit(Exception):
    pass


def ask(prompt):
    value = input(prompt).strip()
    if value.lower() in {"q", "quit", "exit", "退出"}:
        raise UserExit
    if value.lower() in {"b", "back", "返回"}:
        raise PanelBack
    return value
