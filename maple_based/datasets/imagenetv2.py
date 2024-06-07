import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden

from .imagenet import ImageNet
from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNetV2(DatasetBase):
    """ImageNetV2.

    This dataset is used for testing only.
    """

    dataset_dir = "imagenetv2"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        image_dir = "imagenetv2-matched-frequency-format-val"
        self.image_dir = os.path.join(self.dataset_dir, image_dir)

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(root, "imagenet", "split_fewshot", f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                import pickle
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]

        labels = set()
        for item in train:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()

        data = self.read_data(classnames)

        test_base = OxfordPets.subsample_classes(data, labels=labels, subsample='base')
        test_new = OxfordPets.subsample_classes(data, labels=labels, subsample='new')

        super().__init__(train_x=test_base, test_base=test_base, test_new=test_new)

    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = list(classnames.keys())
        items = []

        for label in range(1000):
            class_dir = os.path.join(image_dir, str(label))
            imnames = listdir_nohidden(class_dir)
            folder = folders[label]
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(class_dir, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
