import pandas as pd
import wandb


# data = pd.read_csv("mixup/wandb_export_2022-09-19T17_58_13.059-04_00.csv")


entity, project = "eggroup", "hyperbolic_mixup_1"

columns = [f"{metric}_{i}" for metric in [
    'val_auc', "test_auc", "acc", "sen", "spe"] for i in range(200)]

df = pd.DataFrame(
    columns=["name", "method", "percentage", "augmentation"]+columns)

api = wandb.Api(timeout=600)

runs = api.runs(entity + "/" + project)


i = 0
for r in runs:
    history = r.history()
    summary = r.summary
    config = r.config
    if history.shape[0] == 200:
        t_list = [r.name]

        pass
    # if r.name in names:
    #     if history.shape[0] == 200 or history.shape[0] == 20:
    #         df.loc[i] = name2setting[r.name] + \
    #             history['Test AUC'].tolist()[:20]
    #         i += 1

df.to_csv(f"csv/{entity}_{project}.csv", index=False)

# print(df)
