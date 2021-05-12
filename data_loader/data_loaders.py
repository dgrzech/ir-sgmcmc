from base import BaseDataLoader
from .datasets import BiobankDataset


class BiobankDataLoader(BaseDataLoader):
    def __init__(self, dims, data_dir, save_dirs, sigma_v_init, u_v_init, cps=None):
        self.data_dir, self.save_dirs = data_dir, save_dirs
        self.dataset = BiobankDataset(dims, data_dir, save_dirs, sigma_v_init, u_v_init, cps=cps)

        super().__init__(self.dataset)

    @property
    def dims(self):
        return self.dataset.dims

    @property
    def im_spacing(self):
        return self.dataset.im_spacing

