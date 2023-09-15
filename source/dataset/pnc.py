import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_pnc_data(cfg: DictConfig):

    ts_data = np.load(cfg.dataset.time_seires, allow_pickle=True)
    pearson_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    label_df = pd.read_csv(cfg.dataset.label)

    pearson_data, timeseries_data = pearson_data.item(), ts_data.item()

    pearson_id = pearson_data['id']
    pearson_data = pearson_data['data'][:, :, :]
    id2pearson = dict(zip(pearson_id, pearson_data))

    ts_id = timeseries_data['id']
    timeseries_data = timeseries_data['data']

    id2gender = dict(zip(label_df['SUBJID'], label_df['sex']))

    final_timeseires, final_label, final_pearson = [], [], []

    for fc, l in zip(timeseries_data, ts_id):
        if l in id2gender and l in id2pearson:
            final_timeseires.append(fc)
            final_label.append(id2gender[l])
            final_pearson.append(id2pearson[l])

    final_pearson = np.array(final_pearson)

    final_timeseires = np.array(final_timeseires).transpose(0, 2, 1)

    encoder = preprocessing.LabelEncoder()

    encoder.fit(label_df["sex"])

    labels = encoder.transform(final_label)

    # scaler = StandardScaler(mean=np.mean(
    #     final_timeseires), std=np.std(final_timeseires))

    # final_timeseires = scaler.transform(final_timeseires)

    final_timeseires, final_pearson, labels = [np.array(
        data) for data in (final_timeseires, final_pearson, labels)]

    final_timeseires, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, labels)]

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]
        cfg.dataset.num_classes = labels.unique().shape[0]

    return final_timeseires, final_pearson, labels
