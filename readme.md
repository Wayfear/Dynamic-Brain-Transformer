# Dynamic Brain Transformer

Dynamic Brain Transformer is the open-source implementation of the BHI 2023 paper [Dynamic Brain Transformer with Multi-level Attention for Functional Brain Network Analysis](https://arxiv.org/abs/2309.01941).
---

## Usage

1. Change the *path* attribute in file *source/conf/dataset/PNC.yaml* to the path of your dataset.

2. Run the following command to train the model.

```bash
python -m source --multirun datasz=100p model=gfine dataset=ABCD_reg repeat_time=5 preprocess=non_mixup model.window_sz=360 model.stride=360
```

- **datasz**, default=(10p, 20p, 30p, 40p, 50p, 60p, 70p, 80p, 90p, 100p). How much data to use for training. The value is a percentage of the total number of samples in the dataset. For example, 10p means 10% of the total number of samples in the training set.

- **model**, default=(gfine,). Which model to use. The value is a list of model names. For example, gfine means Dynamic Brain Transformer.

- **dataset**, default=(PNC, ABCD_reg). Which dataset to use. The value is a list of dataset names.

- **repeat_time**, default=5. How many times to repeat the experiment. The value is an integer. For example, 5 means repeat 5 times.

- **preprocess**, default=(mixup, non_mixup). Which preprocess to applied. The value is a list of preprocess names. For example, mixup means mixup, non_mixup means the dataset is feeded into models without preprocess.


## Installation

```bash
conda create --name bnt python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install -c conda-forge wandb
pip install hydra-core --upgrade
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
```

## Citation

Please cite our paper if you find this code useful for your work:
```bibtex
@inproceedings{
  kan2023dbn,
  title={Dynamic Brain Transformer with Multi-level Attention for Functional Brain Network Analysis},
  author={Xuan Kan and Aodong Chen Gu and Hejie Cui and Ying Guo and Carl Yang},
  journal={The IEEE-EMBS International Conference on Biomedical and Health Informatics},
  year={2023},
}
```
