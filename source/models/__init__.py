import imp
from omegaconf import DictConfig
from .gfine import GlobalFineGrinedNN
from .MATT import MixOptimizer


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()
