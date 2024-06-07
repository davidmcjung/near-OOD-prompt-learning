import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

@DATASET_REGISTRY.register()
class Texture(DatasetBase):

    dataset_dir = "dtd"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        data = self.read_data()

        super().__init__(train_x=data, test_base=data)

    def read_data(self,):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        items = []

        for label, classname in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, classname))
            for imname in imnames:
                impath = os.path.join(image_dir, classname, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
