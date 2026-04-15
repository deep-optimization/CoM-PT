<div align="center">
  <h1>🚀 Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models</h1>
  <p>Jiawei Fan, Shigeng Wang, Chao Li, Xiaolong Liu, and Anbang Yao</p>

  [![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](#) 
  [![arXiv](https://img.shields.io/badge/arXiv-2604.12391-b31b1b.svg)](https://arxiv.org/pdf/2604.12391)
</div>

---

This repository contains the official PyTorch implementation of **CoM-PT**. 

### 📢 News
* **[Coming Soon]** ⏳ The evaluation code on clip-benchmark, and more pre-trained VFM model families  will be released shortly. **Watch/Star** this repository to stay updated!
* **[April 2026]** 🎉 We release the training code and pre-trained VFM checkpoints on CC3M dataset. 
* **[Feb 2026]** 🎉 Our paper has been accepted to **CVPR 2026**!

## 🛠️ Installation

```bash
pip install -r requirements-training.txt
pip install -r requirements-test.txt
```

> **Note:** We strongly recommend using `numpy<2.0` in this repository to avoid unnecessary issues during training.

## 🗂️ Dataset Preparation

### Conceptual Captions 3M (CC3M)

OpenCLIP reads a CSV file with two columns: a path to an image, and a text caption. The names of the columns are passed as arguments to `main.py`.

The script `src/data/gather_cc.py` collects the Conceptual Captions 3M images. First, download the [Conceptual Captions 3M URLs](https://ai.google.com/research/ConceptualCaptions/download), and then run the script from our repository. 

For easy notation, we rename `Train_GCC-training` to `cc3m_train`, and `Validation_GCC-1.1.0-Validation` to `cc3m_val`.

```bash
python src/data/gather_cc.py [path/to/cc3m/images/] [path/to/cc3m_train.tsv] [path/to/cc3m_val.tsv]
```

Our downloaded CC3M training set contains 2.89M images, and our CC3M validation set contains 13K images.

> *We also provide a URL where you can directly download the `.zip` file: [Link to zip](#)* 


### Conceptual 12M (CC12M)

The script `src/data/gather_cc12m.py` collects the Conceptual 12M images. First, download the [Conceptual 12M URLs](https://storage.googleapis.com/conceptual_12m/cc12m.tsv), and then run the script from our repository:

```bash
python src/data/gather_cc12m.py [path/to/cc12m/images/] [path/to/cc12m.tsv]
```

> *Since the CC12M dataset is extremely large, the `.zip` file is currently in preparation for release.* 


### Image Descriptions of CC3M and Merged-15M

We do not directly use the generated `cc3m_train.csv` and `cc12m_train.csv` files in our training. Instead, we combine them with MLLM-generated long captions from DreamLIP. You can download `cc3m_lc.csv` and `cc12m_lc.csv` [here](#).

## 🚀 Model Training

Training scripts are provided in the `training_script` folder. Please ensure that the path to the teacher's checkpoint is correctly modified before conducting CoM-PT.

To conduct baseline pre-training:
```bash
bash training_script/cc3m_vit/baseline/baseline_vit-b.sh
```

To conduct CoM-PT:
```bash
bash training_script/cc3m_vit/com-pt/com_vit-s_to_vit-b.sh
```

## 📦 Model Zoo

### ViT Family Pre-trained on the CC3M Dataset

| Network | Method | Train Script | Google Drive |
| :--- | :---: | :---: | :--- |
| ViT-T/16 | Baseline | [`sh`](https://github.com/deep-optimization/CoM-PT/blob/main/training_script/cc3m_vit/baseline/baseline_vit-t.sh) | [baseline_vit-t_e128.pth](https://drive.google.com/file/d/1ixJ0ZZj0-uOKuSWG-zPDAMXNz7wGxHGD/view?usp=drive_link) |
| ViT-S/16 | Baseline | [`sh`](https://github.com/deep-optimization/CoM-PT/blob/main/training_script/cc3m_vit/baseline/baseline_vit-s.sh) | [baseline_vit-s_e128](https://drive.google.com/file/d/1YU0aPimvQYdSB-DGbGeAQ1viS2fW1jSh/view?usp=drive_link) |
| ViT-S/16 | CoM-PT | [`sh`](https://github.com/deep-optimization/CoM-PT/blob/main/training_script/cc3m_vit/com-pt/com_vit-t_to_vit-s.sh) | [com_vit-s_e24.pth](https://drive.google.com/file/d/1BFbSwDg6vrjTDe1zgseCwjH6_2P2CHwt/view?usp=drive_link) |
| ViT-B/16 | Baseline | [`sh`](https://github.com/deep-optimization/CoM-PT/blob/main/training_script/cc3m_vit/baseline/baseline_vit-b.sh) | [baseline_vit-b_e128.pth](https://drive.google.com/file/d/1_6uwor8ESmUgLwqrKKdUIp3VqL-qCa20/view?usp=drive_link) |
| ViT-B/16 | CoM-PT | [`sh`](https://github.com/deep-optimization/CoM-PT/blob/main/training_script/cc3m_vit/com-pt/com_vit-s_to_vit-b.sh) | [com_vit-b_e18.pth](https://drive.google.com/file/d/1-NiUZ2-OWE4wCEVHHCiaXLqWmnloao-8/view?usp=drive_link) |
| ViT-L/16 | Baseline | [`sh`](https://github.com/deep-optimization/CoM-PT/blob/main/training_script/cc3m_vit/baseline/baseline_vit-l.sh) | [baseline_vit-l_e128.pth](https://drive.google.com/file/d/1ixJ0ZZj0-uOKuSWG-zPDAMXNz7wGxHGD/view?usp=drive_link) |
| ViT-L/16 | CoM-PT | [`sh`](https://github.com/deep-optimization/CoM-PT/blob/main/training_script/cc3m_vit/com-pt/com_vit-b_to_vit-l.sh) | [com_vit-l_e15.pth](https://drive.google.com/file/d/1it2LvYm98z03k46mj5pKWePE34UgrZLx/view?usp=drive_link) |

> *More model families are currently being prepared for release.*

## 📊 Model Evaluation

Evaluation on the ImageNet-1K dataset can be performed directly by adding an `--eval` flag to the training scripts.

> *The evaluation on MS-COCO and VTAB+ is built upon `clip-benchmark`, which is in preparation for release.*

## 🙏 Acknowledgement

Our codebase is built upon [open_clip](https://github.com/mlfoundations/open_clip) and [clip-kd](https://github.com/winycg/CLIP-KD). We sincerely thank the authors for releasing their amazing code.

## 📝 Citation

If you find our paper and repository helpful, please consider citing our work:

```bibtex
@inproceedings{fan2026compt,
  title={Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models},
  author={Jiawei Fan, Shigeng Wang, Chao Li, Xiaolong Liu, and Anbang Yao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
