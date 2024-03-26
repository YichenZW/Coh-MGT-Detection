# CoCo: Coherence-Enhanced Machine-Generated Text Detection Under Low Resource With Contrastive Learning

This repo contains code for *CoCo: Coherence-Enhanced Machine-Generated Text Detection Under Low Resource With Contrastive Learning* (https://arxiv.org/abs/2212.10341, EMNLP 2023) by Xiaoming Liu*, Zhaohan Zhang*, Yichen Wang*, Hang Pu, Yu Lan, and Chao Shen. In this codebase, we provide a coherence-graph-based contrastive learning model, CoCo, to detect machine-generated texts under low-resource scenarios. CoCo's detection accuracy outperforms contemporary state-of-art detectors on multiple datasets.

## Data

(1) Install Python 3.9.12 and PyTorch 2.0.1. (slightly older or newer versions are probably also fine for both).

(2) Install other packages by `conda create --name <env> --file requirements.txt`. If it does not work, `pip install -r requirements.txt` is probably also fine; the pip-format `requirements.txt` is in https://github.com/YichenZW/Coh-MGT-Detection/issues/1)

(3) Download our dataset MGTDetect_CoCo from Huggingface (https://huggingface.co/datasets/ZachW/MGTDetect_CoCo) into data/.

* For loading datasets for other usages, we recommend to download the data and using `json.loads()` directly to avoid format errors.

## Preprocess

A two-step preprocess for building the coherence graph for the raw dataset is needed before detection (only for CoCo but not other baseline methods).

(1) Entity extraction: 

`python preprocess/extract_keywords.py --raw_dir data/<dataset_name>.jsonl`. 

A new jsonl file named `data/<dataset_name>_kws.jsonl` will be output under the same path.

(2) Graph construction: 

`python preprocess/construct_graph.py --kw_file_dir data/<dataset_name>_kws.jsonl`. 

The final dataset `data/<dataset_name>_graph.jsonl` will be output under the same path.

## Detection

Before running the code, update the dataset path at `run_detector.py#L1415`.

Example for running the detector with both training, evaluating, and testing with the suggested setting as shown:

`python run_detector.py --args.dataset_name grover_1000` 

* Note that the suggested setting might not be optimal as the based model updating in the future. If the result is obviously low, we suggest you could try `--do_ray True` to tune the hyperparameter again.

### Other Arguments

* Specify `--model_type` and `--model_name_or_path` to use other base models.
* Change `--do_train` and `--do_eval` to False if you have already fine-tuned a model and you want to use the path of it to load it directly.
* Use `--wandb_note` to use wandb and set wandb project name. Must set up wandb at L1188.

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
