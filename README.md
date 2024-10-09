# Near-OOD Prompt Learning
This respository is the official implementation of "[Enhancing Near OOD Detection in Prompt Learning: Maximum Gains, Minimal Costs](https://arxiv.org/abs/2405.16091)". The repository contains models based on [MaPLe repository](https://github.com/muzairkhattak/multimodal-prompt-learning/tree/main) and [KgCoOp repository](https://github.com/htyao89/KgCoOp/tree/main). 

## Data Preparations
Follow the official guidelines of [CoOp](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md) to download
- ImageNet
- Caltech101
- OxfordPets
- StanfordCars
- Flowers102
- Food101
- FGVCAircraft
- SUN397
- DTD
- EuroSAT
- UCF101

and make the `$DATA` directory as
```
$DATA/
|–– imagenet/
|–– caltech-101/
|–– oxford_pets/
|–– stanford_cars/
|–– ...
```
Also, download iNaturalist, SUN, Places, and Texture by following the guidelines of [LoCoOp](https://github.com/AtsuMiyai/LoCoOp?tab=readme-ov-file). Additionaly, download CIFAR10 and CIFAR100 and use this [repository](https://github.com/knjcode/cifar2png) to convert the directory structure as
```
CIFAR10
├── test
│   ├── airplane
│   │    ├──  0001.png
│   │    ├──  0002.png
│   │    ├──  ...
│   ├── automobile
│   ├── ...
└── train
    ├── airplane
    ├── automobile
    ├── ...
```
and save it to `$DATA`.

## Dependencies
Unlike CoOp or MaPLe, we use custom Dassl library. Install PyTorch of a version that meets your local machine and install dependencies by 
```
pip install -r requirements.txt
```

## Scripts
All scripts to reproduce experiments are stored at `scripts/*.sh`.

## Citation
If you are using our work, please cite

```bibtex
@article{jung2024enhancing,
  title={Enhancing Near OOD Detection in Prompt Learning: Maximum Gains, Minimal Costs},
  author={Jung, Myong Chol and Zhao, He and Dipnall, Joanna and Gabbe, Belinda and Du, Lan},
  journal={arXiv preprint arXiv:2405.16091},
  year={2024}
}
```
