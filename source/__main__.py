from datetime import datetime
import wandb
import hydra
from omegaconf import DictConfig, open_dict
from .dataset import dataset_factory
from .models import model_factory
from .components import lr_scheduler_factory, optimizers_factory, logger_factory
from .training import training_factory
from datetime import datetime
from .preprocess import preprocess

# import os
# os.environ['CUDA_VISIBLE_DEVICES']='6'

def model_training(cfg: DictConfig):

    with open_dict(cfg):
        cfg.unique_id = datetime.now().strftime("%m-%d-%H-%M-%S")

    dataloaders = dataset_factory(cfg)
    logger = logger_factory(cfg)

    if cfg.model.name == 'Preprocess':
            obj = preprocess(cfg)
            obj.forward(dataloaders)
            return

    model = model_factory(cfg)
    optimizers = optimizers_factory(model, cfg)
    lr_schedulers = lr_scheduler_factory(lr_configs=cfg.optimizer,
                                         cfg=cfg)
    training = training_factory(cfg, model, optimizers,
                                lr_schedulers, dataloaders, logger)
    
    
    training.train()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    group_name = f"{cfg.dataset.name}_{cfg.model.name}_{cfg.datasz.percentage}_{cfg.preprocess.name}"

    group_name = group_name if "column" not in cfg.dataset else f"{group_name}_{cfg.dataset.column}"

    group_name = group_name if "ts_length" not in cfg.dataset else f"{group_name}_{cfg.dataset.ts_length}"

    group_name = group_name if cfg.name_appendix is None else f"{group_name}_{cfg.name_appendix}"

    group_name = group_name if "alpha" not in cfg.preprocess else f"{group_name}_{cfg.preprocess.alpha}"

    if cfg.group_attr is not None:
        group_name = f'{group_name}_{cfg.group_attr}'

    for _ in range(cfg.repeat_time):
        run = None
        if cfg.logging:
            run = wandb.init(project=cfg.project, entity=cfg.wandb_entity, reinit=True,
                            group=f"{group_name}", tags=[f"{cfg.dataset.name}"])
        
            wandb.config.update({
                "group_name": group_name,
                "dataset": cfg.dataset.name,
                "model": cfg.model.name,
                "datasz": cfg.datasz.percentage,
                "preprocess": cfg.preprocess.name,
                "column": cfg.dataset.column if "column" in cfg.dataset else None,
                "ts_length": cfg.dataset.ts_length if "ts_length" in cfg.dataset else None,
                "alpha": cfg.preprocess.alpha if "alpha" in cfg.preprocess else None,
            })
        model_training(cfg)

        if cfg.logging:
            run.finish()


if __name__ == '__main__':
    main()
