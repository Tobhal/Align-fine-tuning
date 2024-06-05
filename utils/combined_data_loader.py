from torch.utils.data.dataloader import DataLoader

class CombinedLoader:
    def __init__(self, loader1: DataLoader, loader2: DataLoader):
        self.loader1 = loader1
        self.loader2 = loader2

    def __iter__(self):
        self.loader1_iter = iter(self.loader1)
        self.loader2_iter = iter(self.loader2)
        return self

    def __next__(self):
        try:
            data = next(self.loader1_iter)
        except StopIteration:
            data = next(self.loader2_iter)
        return data
    
    def __len__(self):
        return len(self.loader1) + len(self.loader2)