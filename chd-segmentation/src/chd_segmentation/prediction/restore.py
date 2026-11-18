def restore(probabilities, entry, plan):
    from .geometry import restore_source
    return restore_source(probabilities, entry)
