import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}
BASE = ['accordion',
 'airplane',
 'anchor',
 'ant',
 'barrel',
 'bass',
 'beaver',
 'binocular',
 'bonsai',
 'brain',
 'brontosaurus',
 'buddha',
 'butterfly',
 'camera',
 'cannon',
 'car_side',
 'ceiling_fan',
 'cellphone',
 'chair',
 'chandelier',
 'cougar_face',
 'crab',
 'crayfish',
 'crocodile',
 'dalmatian',
 'dolphin',
 'dragonfly',
 'electric_guitar',
 'elephant',
 'emu',
 'euphonium',
 'ewer',
 'face',
 'ferry',
 'flamingo',
 'garfield',
 'gerenuk',
 'gramophone',
 'inline_skate',
 'kangaroo',
 'ketch',
 'leopard',
 'llama',
 'lobster',
 'lotus',
 'motorbike',
 'nautilus',
 'octopus',
 'pizza',
 'stapler']

NEW = ['cougar_body',
 'crocodile_head',
 'cup',
 'dollar_bill',
 'flamingo_head',
 'grand_piano',
 'hawksbill',
 'headphone',
 'hedgehog',
 'helicopter',
 'ibis',
 'joshua_tree',
 'lamp',
 'laptop',
 'mandolin',
 'mayfly',
 'menorah',
 'metronome',
 'minaret',
 'okapi',
 'pagoda',
 'panda',
 'pigeon',
 'platypus',
 'pyramid',
 'revolver',
 'rhino',
 'rooster',
 'saxophone',
 'schooner',
 'scissors',
 'scorpion',
 'sea_horse',
 'snoopy',
 'soccer_ball',
 'starfish',
 'stegosaurus',
 'stop_sign',
 'strawberry',
 'sunflower',
 'tick',
 'trilobite',
 'umbrella',
 'watch',
 'water_lilly',
 'wheelchair',
 'wild_cat',
 'windsor_chair',
 'wrench',
 'yin_yang']

@DATASET_REGISTRY.register()
class Caltech101(DatasetBase):

    dataset_dir = "caltech-101"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.split_path):
            train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)
        else:
            train, val, test = DTD.read_and_split_data(self.image_dir, ignored=IGNORED, new_cnames=NEW_CNAMES)
            OxfordPets.save_split(train, val, test, self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        base_labels = set()
        new_labels = set()
        for item in train:
            if item.classname in BASE:
                base_labels.add(item.label)
            elif item.classname in NEW:
                new_labels.add(item.label)
        base_labels = list(base_labels)
        new_labels = list(new_labels)
        #labels.sort()
        if cfg.DATASET.SUBSAMPLE_CLASSES == 'base':
            labels = base_labels
        elif cfg.DATASET.SUBSAMPLE_CLASSES == 'new':
            labels = new_labels
        else:
            labels = base_labels + new_labels
        train = OxfordPets.subsample_classes(train, labels=labels, subsample=cfg.DATASET.SUBSAMPLE_CLASSES, custom=True)
        val_base, test_base = OxfordPets.subsample_classes(val, test, labels=base_labels, subsample='base', custom=True)
        val_new, test_new = OxfordPets.subsample_classes(val, test, labels=new_labels, subsample='new', custom=True)
        super().__init__(train_x=train, val_base=val_base, val_new=val_new, test_base=test_base, test_new=test_new)
