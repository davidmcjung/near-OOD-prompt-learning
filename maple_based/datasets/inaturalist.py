import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

@DATASET_REGISTRY.register()
class iNaturalist(DatasetBase):

    dataset_dir = "iNaturalist"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        data = self.read_data()

        super().__init__(train_x=data, test_base=data)

    def read_data(self,):
        image_dir = self.image_dir
        items = []

        imnames = listdir_nohidden(image_dir)
        for imname in imnames:
            impath = os.path.join(image_dir, imname)
            item = Datum(impath=impath, label=-1, classname='OOD')
            items.append(item)

        return items
