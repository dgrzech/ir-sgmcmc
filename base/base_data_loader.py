from torch.utils.data import DataLoader


class BaseDataLoader(DataLoader):
    """
    base class for all data loaders
    """

    def __init__(self, dataset):
        init_kwargs = {'dataset': dataset, 'pin_memory': True}
        super().__init__(sampler=None, **init_kwargs)
