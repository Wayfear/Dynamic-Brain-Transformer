import torch
import torch.utils.data as utils
import numpy as np
import source.dataset as dataset
from torch.distributions import normal
from scipy.stats import multivariate_normal


def estimate_edge_distribution_for_each_class(dataloader: utils.DataLoader):
    pearsons, labels = [], []
    for _, pearson, label in dataloader:
        pearsons.append(pearson)
        labels.append(label)

    pearsons = torch.cat(pearsons)
    node_sz = pearsons.shape[1]
    index = np.triu_indices(node_sz, k=1)
    pearsons = pearsons[:, index[0], index[1]]
    labels = torch.argmax(torch.cat(labels), dim=-1)

    label2mean_std = {}

    for label in labels.unique():

        mean = torch.mean(pearsons[labels == label], dim=0)
        std = torch.std(pearsons[labels == label], dim=0)
        label2mean_std[label.item()] = {"mean": mean, "std": std}

    return label2mean_std, index, node_sz


def graphon_mixup(alpha: int = 1.0):
    label2mean_std, index, node_sz = dataset.edge_weight_distribution

    def sample_data(sample_num, y):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        label_y = torch.argmax(y, dim=-1)
        unique_y = torch.unique(label_y).tolist()
        if len(unique_y) == 1:
            used_cls = [unique_y[0], unique_y[0]]
        else:
            used_cls = np.random.choice(unique_y, 2, replace=False)

        mean = lam*label2mean_std[used_cls[0]]['mean'] + \
            (1-lam)*label2mean_std[used_cls[1]]['mean']
        std = lam*label2mean_std[used_cls[0]]['std'] + \
            (1-lam)*label2mean_std[used_cls[1]]['std']
        generator = normal.Normal(mean, std)
        samples = generator.sample([sample_num])

        new_sample = torch.ones((sample_num, node_sz, node_sz))
        label = torch.zeros_like(y)
        label[:, used_cls[0]] = lam
        label[:, used_cls[1]] = 1-lam
        new_sample[:, index[0], index[1]] = samples
        new_sample[:, index[1], index[0]] = samples

        return new_sample, label

    return sample_data


def estimate_edge_distribution_for_regression(dataloader: utils.DataLoader):
    pearsons, labels = [], []
    for _, pearson, label in dataloader:
        pearsons.append(pearson)
        labels.append(label)

    pearsons = torch.cat(pearsons)
    node_sz = pearsons.shape[1]
    index = np.triu_indices(node_sz, k=1)
    pearsons = pearsons[:, index[0], index[1]]
    labels = torch.cat(labels)[:, 0]
    variable_num = pearsons.shape[1]
    coef = np.zeros(variable_num)
    for i in range(variable_num):
        coef[i] = np.corrcoef(pearsons[:, i], labels)[0, 1]
        if i % 1000 == 0:
            print(i)
    coef = torch.FloatTensor(coef)
    edge_mean = torch.mean(pearsons, dim=0)
    edge_std = torch.std(pearsons, dim=0)
    y_mean = torch.mean(labels, dim=0)
    y_std = torch.std(labels, dim=0)
    return coef, edge_mean, edge_std, y_mean, y_std, index, node_sz


def graphon_mixup_for_regression(alpha: int = 1.0):
    """
    cond_multivariate_normal_distribution
    """
    coef, edge_mean, edge_std, y_mean, y_std, index, node_sz = dataset.edge_weight_distribution

    def sample_data(sample_num, y):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        ramdom_index = torch.randperm(sample_num)

        y = lam * y + (1-lam) * y[ramdom_index]

        mean = edge_mean + (1.0/y_std)*edge_std*coef*(y-y_mean)
        std = (1-torch.square(coef))*torch.square(edge_std)
        std = std.repeat(sample_num, 1)

        generator = normal.Normal(mean, std)
        samples = generator.sample([1])[0]

        new_sample = torch.ones((sample_num, node_sz, node_sz))
        new_sample[:, index[0], index[1]] = samples
        new_sample[:, index[1], index[0]] = samples

        return new_sample, y

    return sample_data
