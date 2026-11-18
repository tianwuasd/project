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
