import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_abcd_data(cfg: DictConfig):

    ts_data = np.load(cfg.dataset.time_seires, allow_pickle=True)
    pearson_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    label_df = pd.read_csv(cfg.dataset.label)

    label_df = label_df[['id', cfg.dataset.column]]
    label_df = label_df[~label_df[cfg.dataset.column].isnull()]

    with open(cfg.dataset.node_id, 'r') as f:
        lines = f.readlines()
        pearson_id = [line[:-1] for line in lines]

    with open(cfg.dataset.seires_id, 'r') as f:
        lines = f.readlines()
        ts_id = [line[:-1] for line in lines]

    id2pearson = dict(zip(pearson_id, pearson_data))

    id2label = dict(zip(label_df['id'], label_df[cfg.dataset.column]))

    final_timeseires, final_label, final_pearson = [], [], []

    for ts, l in zip(ts_data, ts_id):
        if l in id2label and l in id2pearson:
            if np.any(np.isnan(id2pearson[l])) == False:
                final_timeseires.append(ts)
                final_label.append(id2label[l])
                final_pearson.append(id2pearson[l])

    if not cfg.dataset.regression:

        encoder = preprocessing.LabelEncoder()

        encoder.fit(label_df[cfg.dataset.column])

        labels = encoder.transform(final_label)

    else:
        labels = final_label

    final_timeseires, final_pearson, labels = [np.array(
        data) for data in (final_timeseires, final_pearson, labels)]

    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]
        cfg.dataset.num_classes = labels.unique().shape[0]
    if "stratified" in cfg.dataset and cfg.dataset.stratified:
        return final_timeseires, final_pearson, labels, labels
    return final_timeseires, final_pearson, labels
