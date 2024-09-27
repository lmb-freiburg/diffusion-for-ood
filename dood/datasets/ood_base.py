from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class BaseOoDDataset(BaseSegDataset):
    def __init__(self, ood_index, **kwargs):
        super().__init__(**kwargs)
        self.ood_index = ood_index