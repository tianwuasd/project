def require_complete_epoch(*,steps,limit,batches):
    if batches<1:
        raise ValueError('MedSAM2当前训练名单不足以形成一个实际batch')
    if limit is not None and steps<limit<steps+batches:
        raise ValueError('MedSAM2停止位置必须位于完整官方epoch边界')


class BoundedLoader:
    def __init__(self, loader, count, claim, finish=None):
        self.loader, self.count, self.claim = loader, min(len(loader), count), claim
        self.finish = finish
        self.dataloaders = loader.dataloaders

    def __len__(self):
        return self.count

    def __iter__(self):
        iterator = iter(self.loader)
        try:
            for _ in range(self.count):
                self.claim()
                yield next(iterator)
                if self.finish is not None:
                    self.finish()
        finally:
            iterator.close()
