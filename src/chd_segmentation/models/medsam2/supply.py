from time import perf_counter

from torch.utils.data import Dataset

from ...observability.events import event


class TaggedBatches:
    def __init__(self,batches):
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch,indices in enumerate(self.batches):
            yield [(batch,position,index) for position,index in enumerate(indices)]


class OrderedDataset(Dataset):
    def __init__(self,dataset,supply,rank):
        self.dataset,self.supply,self.rank = dataset,supply,rank

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,ticket):
        batch,position,index = ticket
        with self.supply.build(self.rank,batch,position):
            from .data import sample_supply
            sample_supply(self.supply,self.rank)
            return self.dataset[index]


class ConsumedLoader:
    def __init__(self,loader,supply,rank,epoch,device):
        if len(loader.dataloaders)!=1:
            raise ValueError('先心MedSAM2供数需要单一官方心脏数据集')
        self.loader,self.supply,self.rank = loader,supply,rank
        self.epoch,self.device = epoch,device
        self.dataloaders = loader.dataloaders

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch,value in enumerate(self.loader):
            self.supply.ready.wait(self.rank,f'med-data:{self.epoch}:{batch}')
            started = perf_counter()
            yield value
            self.supply.finish(self.rank,f'med-epoch:{self.epoch}',self.device)
            event(self.supply.root,'model_batch_consumed',rank=self.rank,batch=batch,epoch=self.epoch,
                  seconds=perf_counter()-started,device_completion_checked=True)
            del value
