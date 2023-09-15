import imp
from omegaconf import DictConfig
from .gfine import GlobalFineGrinedNN
from .st_gcn_net import st_gcn_tony
from .stagin import staginWarp
from .MATT import MixOptimizer


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()
