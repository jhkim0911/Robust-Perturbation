![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)
# IEEE Transactions on Image Processing
[![Generic badge](https://img.shields.io/badge/Library-Pytorch-green.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/DooDooKim/Robust-Perturbation/master/LICENSE) 

## Robust Perturbation for Visual Explanation: Cross-Checking Mask Optimization to Avoid Class Distortion

#### Authors: [Junho Kim<sup>1</sup>](https://scholar.google.com/citations?user=ZxE16ZUAAAAJ&hl=en), [Seongyeop Kim<sup>1</sup>](https://scholar.google.com/citations?user=gpX2FNEAAAAJ&hl=ko&oi=ao), [Seong Tae Kim<sup>2</sup>](https://scholar.google.co.kr/citations?user=DEQbVOwAAAAJ&hl=ko), and [Yong Man Ro<sup>1</sup>](https://scholar.google.co.kr/citations?user=IPzfF7cAAAAJ&hl=en)
##### Affiliation: School of Electrical Engineering, Korea Advanced Institute of Science and Technology (KAIST)<sup>1</sup>, Department of Computer Science and Engineering, Kyung Hee University (KHU) <sup>2</sup>
##### Email: `arkimjh@kaist.ac.kr`, `seongyeop@kaist.ac.kr`, `st.kim@khu.ac.kr`, `ymro@kaist.ac.kr`

---

> 
This repository contains the official PyTorch implementation of the following paper:
> **Robust Perturbation for Visual Explanation: Cross-Checking Mask Optimization to Avoid Class Distortion" published in IEEE Transactions on Image Processing (TIP)**<br>
> Paper: https://ieeexplore.ieee.org/abstract/document/9633238<br>
> 
> **Abstract** *Along with the outstanding performance of the deep neural networks (DNNs), considerable research efforts have been devoted to finding ways to understand the decision of DNNs structures. In the computer vision domain, visualizing the attribution map is one of the most intuitive and understandable ways to achieve human-level interpretation. Among them, perturbation-based visualization can explain the “black box” property of the given network by optimizing perturbation masks that alter the network prediction of the target class the most. However, existing perturbation methods could make unexpected changes to network predictions after applying a perturbation mask to the input image, resulting in a loss of robustness and fidelity of the perturbation mechanisms. In this paper, we define class distortion as the unexpected changes of the network prediction during the perturbation process. To handle that, we propose a novel visual interpretation framework, Robust Perturbation, which shows robustness against the unexpected class distortion during the mask optimization. With a new cross-checking mask optimization strategy, our proposed framework perturbs the target prediction of the network while upholding the non-target predictions, providing more reliable and accurate visual explanations. We evaluate our framework on three different public datasets through extensive experiments. Furthermore, we propose a new metric for class distortion evaluation. In both quantitative and qualitative experiments, tackling the class distortion problem turns out to enhance the quality and fidelity of the visual explanation in comparison with the existing perturbation-based methods.*



<p align="center">
  <img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/83/9626658/9633238/ro5abcdefghijklmn-3130526-large.gif" width="700px"/></p>

---
### Citation
If you find this work helpful, please cite it as:

```
@article{kim2021robust,
  title={Robust Perturbation for Visual Explanation: Cross-Checking Mask Optimization to Avoid Class Distortion},
  author={Kim, Junho and Kim, Seongyeop and Kim, Seong Tae and Ro, Yong Man},
  journal={IEEE Transactions on Image Processing},
  volume={31},
  pages={301--313},
  year={2021},
  publisher={IEEE}
}
```
---

### Datasets and Baseline Models
* [PASCAL VOC 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
* [VGG-16](https://arxiv.org/abs/1409.1556)
* [GoogLeNet](https://arxiv.org/abs/1409.4842)
* [ResNet-50](https://arxiv.org/abs/1512.03385)

---

### Training Model
The weights are saved in `--result_dir` and shows the training logs in `--log_dir`.

To train the model from scratch, run following command:
```shell
python main.py \
--phase train \
--model_name ResNet \
--dataset VOC \
--epoch 100 --batch_size 128 --lr 0.0001 \
--device cuda:0 --print_freq 50
```

---

### Testing Model
To test the model, run following command:
```shell
python main.py --phase test --model_name ResNet --dataset VOC
```

---

### Generating Attribution maps
To generate attribution map of Robust Perturbation, run following command:
```shell
python main.py --phase rp --model_name ResNet --dataset VOC
```

---

### Evaluation for Attribution maps
Several evaluation metrics you can validate:

Energy-based Pointing Game [(paper)](https://arxiv.org/abs/1910.01279)
```shell
python main.py --phase energy --model_name ResNet --dataset VOC
```

Class Distortion Score
```shell
python main.py --phase c_sen --model_name ResNet --dataset VOC
```

Insertion & Deletion Game [(paper)](https://arxiv.org/abs/1806.07421)
```shell
python main.py --phase indel --model_name ResNet --dataset VOC
```
