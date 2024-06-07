import os
import pickle
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


@DATASET_REGISTRY.register()
class Cifar10(DatasetBase):

    dataset_dir = "CIFAR10"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            train = self.read_data("train")
            test = self.read_data("test")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        #labels = set()
        #for item in train:
        #    labels.add(item.label)
        #labels = list(labels)
        #labels.sort()
        #train = OxfordPets.subsample_classes(train, labels=labels, subsample=cfg.DATASET.SUBSAMPLE_CLASSES)
        #test_base = OxfordPets.subsample_classes(test, labels=labels, subsample='base')
        #test_new = OxfordPets.subsample_classes(test, labels=labels, subsample='new')
        super().__init__(train_x=train, test_base=test, test_new=test)

    def read_data(self, split_dir):
        split_dir = os.path.join(self.dataset_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for label, classname in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, classname))
            for imname in imnames:
                impath = os.path.join(split_dir, classname, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items