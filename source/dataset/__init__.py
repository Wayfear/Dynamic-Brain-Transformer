from omegaconf import DictConfig, open_dict
from .abcd import load_abcd_data
from .abide import load_abide_data
from .pnc import load_pnc_data
from .tcga import load_tcga_data
from .dataloader import init_dataloader, init_stratified_dataloader
from typing import List
import torch.utils as utils
from source.utils import obtain_partition, estimate_edge_distribution_for_each_class, estimate_edge_distribution_for_regression


def dataset_factory(cfg: DictConfig) -> List[utils.data.DataLoader]:

    assert cfg.dataset.name in ['abcd', 'abide', 'pnc', 'tcga']

    datasets = eval(
        f"load_{cfg.dataset.name}_data")(cfg)

    dataloaders = init_stratified_dataloader(cfg, *datasets) \
        if cfg.dataset.stratified \
        else init_dataloader(cfg, *datasets)

    if cfg.preprocess.graphon:
        global edge_weight_distribution
        if cfg.dataset.regression:
            edge_weight_distribution = estimate_edge_distribution_for_regression(
                dataloaders[0])
        else:
            edge_weight_distribution = estimate_edge_distribution_for_each_class(
                dataloaders[0])

    if cfg.model.name == "SparseNN":
        global masks_from_data
        masks_from_data = obtain_partition(
            dataloaders[0], cfg.model.threshold, cfg.model.step)
    return dataloaders
