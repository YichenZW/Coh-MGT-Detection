# CoCo: Coherence-Enhanced Machine-Generated Text Detection Under Low Resource With Contrastive Learning

This repo contains code for *CoCo: Coherence-Enhanced Machine-Generated Text Detection Under Low Resource With Contrastive Learning* (https://arxiv.org/abs/2212.10341, EMNLP 2023) by Xiaoming Liu*, Zhaohan Zhang*, Yichen Wang*, Hang Pu, Yu Lan, and Chao Shen. In this codebase we provide a coherence-graph-based contrastive learning model, CoCo, to detect machine-generated texts under low-resource scenario. CoCo's detection accuracy outperforms the state-of-art detectors on multiple datasets.

## Data

(1) Install Python 3.9.12 and PyTorch 2.0.1. (slightly older or newer versions are probably also fine for both).

(2) Install other packages by `conda create --name <env> --file requirements.txt`. (`pip install -r requirements.txt` are probably also fine)

(3) Download our dataset MGTDetect_CoCo from Huggingface (https://huggingface.co/datasets/ZachW/MGTDetect_CoCo) into data/.

## Preprocess

[TBD]

## Detection

[TBD]

## Citation

If you find our work helpful, please cite us with the following BibTex entry:

```
@article{liu2022coco,
  title={Coco: Coherence-enhanced machine-generated text detection under data limitation with contrastive learning},
  author={Liu, Xiaoming and Zhang, Zhaohan and Wang, Yichen and Pu, Hang and Lan, Yu and Shen, Chao},
  journal={arXiv preprint arXiv:2212.10341},
  year={2022}
}
```

Link to EMNLP 2023 version paper on ACL Anthology: https://aclanthology.org/2023.emnlp-main.1005/ 