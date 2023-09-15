import numpy as np
from omegaconf import DictConfig, open_dict
from sklearn import preprocessing
import torch
from collections import Counter


def load_tcga_data(cfg: DictConfig):

    loaded_data = np.load(cfg.dataset.node_feature, allow_pickle=True)

    loaded_data = loaded_data.item()
    pearson_data = loaded_data['data']
    pearson_data = pearson_data[:, :15, :15]
    label = loaded_data['labels']

    c = Counter(label)

    selected_samples = []

    for i, (l, v) in enumerate(c.most_common()):
        if i >= cfg.dataset.num_classes:
            break
        index = np.where(np.char.equal(label, l))[0]
        selected_samples.append(index)

    selected_samples = np.concatenate(selected_samples)

    np.random.shuffle(selected_samples)

    encoder = preprocessing.LabelEncoder()
    encoder.fit(label)
    label = encoder.transform(label)

    pearson_data = pearson_data[selected_samples]
    label = label[selected_samples]

    encoder = preprocessing.LabelEncoder()
    encoder.fit(label)
    label = encoder.transform(label)

    pearson_data, label = [torch.from_numpy(
        data).float() for data in (pearson_data, label)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = pearson_data.shape[1:]
        cfg.dataset.timeseries_sz = 0
        cfg.dataset.num_classes = label.unique().shape[0]

    if "stratified" in cfg.dataset and cfg.dataset.stratified:
        return pearson_data, pearson_data, label, label
    return pearson_data, pearson_data, label
