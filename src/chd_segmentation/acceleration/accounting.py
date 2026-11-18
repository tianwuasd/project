def reconcile(state, rss, fixed_rss):
    # 只将申请之后采样到的工作内存增长抵扣预留；新worker的启动基数单独计入。
    working = max(0,rss-fixed_rss)
    change = working-state.get('claim_working_rss',working)
    claims = state.get('claims',{})
    for token in list(claims) if change >= 0 else reversed(list(claims)):
        claim = claims[token]
        amount = min(change,claim['amount']-claim['realized']) if change >= 0 else -min(-change,claim['realized'])
        claim['realized'] += amount
        change -= amount
    state['claims'] = claims
    state['claim_working_rss'] = working
    state['unrealized_bytes'] = sum(row['amount']-row['realized'] for row in claims.values())


def add_claim(state, token, amount, kind):
    claims = state['claims']
    claims[token] = dict(amount=amount,realized=0,kind=kind)
    state['claims'] = claims
    state['unrealized_bytes'] += amount


def remove_claim(state, token):
    claims = state['claims']
    row = claims.pop(token)
    state['claims'] = claims
    state['unrealized_bytes'] -= row['amount']-row['realized']


def available(state, *, protect_pool=False):
    idle_pool = max(0,state.get('workspace_capacity',0)-state.get('workspace_reserved',0)) if protect_pool else 0
    free = state['capacity']-state.get('task_rss',0)-state.get('unrealized_bytes',0)-idle_pool
    return max(0,min(state['headroom'],free))
