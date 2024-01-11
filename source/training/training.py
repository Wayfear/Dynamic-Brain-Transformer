from source.utils import accuracy, TotalMeter, count_params, isfloat
import torch
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, classification_report
from source.utils import continus_mixup_data, discrete_mixup_data, graphon_mixup, graphon_mixup_for_regression, renn_mixup, get_batch_kde_mixup_batch, drop_node, drop_edge
import wandb
from omegaconf import DictConfig
from typing import List
import torch.utils.data as utils
from source.components import LRScheduler
import logging
import pandas as pd
import matplotlib.pyplot as plt
from random import randrange


class Train:

    def __init__(self, cfg: DictConfig,
                 model: torch.nn.Module,
                 optimizers: List[torch.optim.Optimizer],
                 lr_schedulers: List[LRScheduler],
                 dataloaders: List[utils.DataLoader],
                 logger: logging.Logger) -> None:

        self.config = cfg
        self.logger = logger
        self.model = model
        self.logger.info(
            f'#model params: {count_params(self.model)}, sample size: {cfg.dataset.sample_size}')
        self.train_dataloader, self.val_dataloader, self.test_dataloader = dataloaders
        self.epochs = cfg.training.epochs
        self.total_steps = cfg.total_steps
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        if cfg.dataset.regression:
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
        self.save_path = Path(cfg.log_path) / cfg.unique_id
        self.save_learnable_graph = cfg.save_learnable_graph
        if cfg.preprocess.graphon:
            if cfg.dataset.regression:
                self.sampler = graphon_mixup_for_regression()
            else:
                self.sampler = graphon_mixup()

        self.init_meters()

    def init_meters(self):
        self.train_loss, self.val_loss,\
            self.test_loss, self.train_accuracy,\
            self.val_accuracy, self.test_accuracy = [
                TotalMeter() for _ in range(6)]

    def reset_meters(self):
        for meter in [self.train_accuracy, self.val_accuracy,
                      self.test_accuracy, self.train_loss,
                      self.val_loss, self.test_loss]:
            meter.reset()

    def train_per_epoch(self, optimizer, lr_scheduler):
        self.model.train()
        a, sp, v = None, None, None
        
        for time_series, node_feature, label in self.train_dataloader:
            if self.config.model.name == 'staginWarp':
                time_series = time_series.transpose(1, 2)
                a, sp, v = self.staginWarpInit(time_series, v)
            
            label = label.float()
            self.current_step += 1

            if torch.isnan(time_series).any():
                pass

            lr_scheduler.update(optimizer=optimizer, step=self.current_step)

            if self.config.preprocess.discrete:
                node_feature, label = discrete_mixup_data(
                    node_feature, label)
            if self.config.preprocess.continus:
                time_series, node_feature, label = continus_mixup_data(
                    time_series, node_feature, y=label)
            if self.config.preprocess.graphon:
                new_node_feature, new_label = self.sampler(
                    sample_num=label.shape[0], y=label)
                node_feature = torch.concat((node_feature, new_node_feature))
                label = torch.concat((label, new_label))
            if self.config.preprocess.renn_mixup:
                node_feature, label = renn_mixup(node_feature, label, self.config.preprocess.alpha)
            if self.config.preprocess.c_mixup:
                node_feature, label = get_batch_kde_mixup_batch(
                    node_feature, label)
            if self.config.preprocess.drop_edge:
                node_feature = drop_edge(node_feature)
            if self.config.preprocess.drop_node:
                node_feature = drop_node(node_feature)  
            try:
                predict = None
                if self.config.model.name == 'staginWarp':
                    predict = self.model(a, v, sp, time_series)
                    label = label.cuda()
                else:
                    time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
                    predict = self.model(time_series, node_feature)

                loss = self.loss_fn(predict, label)
                if torch.isnan(loss).any():
                    print("Nan in loss")
                self.train_loss.update_with_weight(loss.item(), label.shape[0])
                optimizer.zero_grad()
                try:
                    loss.backward()
                except Exception as back_err:
                    optimizer.zero_grad()
                    print(f'backward error: {back_err}')
                    pass
                optimizer.step()
                if not self.config.dataset.regression:
                    top1 = accuracy(predict, label[:, 1])[0]
                    self.train_accuracy.update_with_weight(top1, label.shape[0])
            except Exception as for_err:
                optimizer.zero_grad()
                print(f'forward error: {for_err}')
                pass
            # wandb.log({"LR": lr_scheduler.lr,
            #            "Iter loss": loss.item()})

    def test_per_epoch(self, dataloader, loss_meter, acc_meter):
        labels = []
        result = []

        self.model.eval()
        a, sp, v = None, None, None
        with torch.no_grad():
            for time_series, node_feature, label in dataloader:
                output = None
                if self.config.model.name == 'staginWarp':
                    time_series = time_series.transpose(1, 2)
                    a, sp, v = self.staginWarpInit(time_series, v)
                    output = self.model(a, v, sp, time_series)
                    label = label.cuda()
                else:
                    time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
                    output = self.model(time_series, node_feature)

                label = label.float()

                loss = self.loss_fn(output, label)
                loss_meter.update_with_weight(
                    loss.item(), label.shape[0])
                # optimizer.zero_grad()
                if not self.config.dataset.regression:
                    top1 = accuracy(output, label[:, 1])[0]
                    acc_meter.update_with_weight(top1, label.shape[0])

                    if self.config.dataset.num_classes == 2:
                        result += F.softmax(output, dim=1)[:, 1].tolist()
                        labels += label[:, 1].tolist()
                    else:
                        result.append(output.detach().cpu())
                        labels.append(label.detach().cpu())

        if self.config.dataset.regression:
            return None

        if self.config.dataset.num_classes == 2:

            auc = roc_auc_score(labels, result)
            result, labels = np.array(result), np.array(labels)
            result[result > 0.5] = 1
            result[result <= 0.5] = 0
            metric = precision_recall_fscore_support(
                labels, result, average='micro')

            report = classification_report(
                labels, result, output_dict=True, zero_division=0)

            recall = [0, 0]
            for k in report:
                if isfloat(k):
                    recall[int(float(k))] = report[k]['recall']
            return [auc] + list(metric) + recall

        else:

            result, labels = torch.vstack(result), torch.vstack(labels)
            labels = torch.argmax(labels, dim=1)
            result = torch.argmax(result, dim=1)
            # top1, top3, top5 = accuracy(result, labels, top_k=(1, 3, 5))
            metric = precision_recall_fscore_support(
                labels, result, average='macro')
            return list(metric)

    def generate_save_learnable_matrix(self):

        # wandb.log({'heatmap_with_text': wandb.plots.HeatMap(x_labels, y_labels, matrix_values, show_text=False)})
        learable_matrixs = []

        labels = []

        for time_series, node_feature, label in self.test_dataloader:
            label = label.long()
            time_series, node_feature, label = time_series.cuda(), node_feature.cuda(), label.cuda()
            _, learable_matrix, _ = self.model(time_series, node_feature)

            learable_matrixs.append(learable_matrix.cpu().detach().numpy())
            labels += label.tolist()

        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"learnable_matrix.npy", {'matrix': np.vstack(
            learable_matrixs), "label": np.array(labels)}, allow_pickle=True)
    
    def save_attention(self, step):
        grind_matrixs = []
        #global_matrixs = []
        size = 0
        for time_series, node_feature, _ in self.test_dataloader:
            time_series, node_feature = time_series.cuda(), node_feature.cuda()
            grind_matrix = self.model(time_series, node_feature, True)
            grind_matrixs.append(np.sum(grind_matrix.cpu().detach().numpy(), axis = 0))
            #global_matrixs.append(np.sum(global_matrix.cpu().detach().numpy(), axis = 0))
            size += 1
        for time_series, node_feature, _ in self.train_dataloader:
            time_series, node_feature = time_series.cuda(), node_feature.cuda()
            grind_matrix = self.model(time_series, node_feature, True)
            grind_matrixs.append(np.sum(grind_matrix.cpu().detach().numpy(), axis = 0))
            #global_matrixs.append(np.sum(global_matrix.cpu().detach().numpy(), axis = 0))
            size += 1
        grind_matrixs = np.squeeze(np.dstack(grind_matrixs))
        #global_matrixs = np.squeeze(np.dstack(global_matrixs))
        mask = self.model.getMask().cpu().detach().numpy()
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"attention_matrix.npy", {"grind": np.mean(grind_matrixs, axis= 1),
                                                        #"global": np.mean(global_matrixs, axis= 1),
                                                        "mask": mask}, allow_pickle=True)
        # need to specify the node name and category to plot
        # commented for now
        #fig1 = self.graph_heatMap(mask)
        #fig2 = self.graph_linePlot(grind_matrixs)
        #fig3 = self.graph_linePlot(global_matrixs)
        #wandb.log({"mask": fig1, "grind_matrixs": wandb.Image(fig2)}, step = step)
        plt.close('all')
        

    def save_result(self, results: torch.Tensor):
        self.save_path.mkdir(exist_ok=True, parents=True)
        np.save(self.save_path/"training_process.npy",
                results, allow_pickle=True)

        self.save_model(f'{self.test_loss.avg: .3f}')

    def save_model(self, comments: str):
        torch.save(self.model.state_dict(), self.save_path/f'model_{comments}.pt')

    def train(self):
        training_process = []
        self.current_step = 0
        for epoch in range(self.epochs):
            self.reset_meters()
            self.train_per_epoch(self.optimizers[0], self.lr_schedulers[0])
            torch.cuda.empty_cache()
            # if epoch % 100 == 0 and epoch != 0:
            #     val_result = self.test_per_epoch(self.val_dataloader,
            #                                  self.val_loss, self.val_accuracy)

            #     test_result = self.test_per_epoch(self.test_dataloader,
            #                                     self.test_loss, self.test_accuracy)
            # else:
            #     continue
            val_result = self.test_per_epoch(self.val_dataloader,
                                             self.val_loss, self.val_accuracy)

            test_result = self.test_per_epoch(self.test_dataloader,
                                              self.test_loss, self.test_accuracy)
            if self.config.model.save_graph and epoch % 10 == 0:
                self.save_attention(epoch)

            if self.config.dataset.regression:

                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .3f}',
                    f'Val Loss:{self.val_loss.avg: .3f}',
                    f'Test Loss:{self.test_loss.avg: .3f}',
                ]))
                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Val Loss": self.val_loss.avg,
                    "Test Loss": self.test_loss.avg,
                })
                training_process.append({
                    "Epoch": epoch,
                    "Train Loss": self.train_loss.avg,
                    "Test Loss": self.test_loss.avg,
                    "Val Loss": self.val_loss.avg,
                })

            elif self.config.dataset.num_classes == 2:

                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .3f}',
                    f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                    f'Test Loss:{self.test_loss.avg: .3f}',
                    f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                    f'Val AUC:{val_result[0]:.4f}',
                    f'Test AUC:{test_result[0]:.4f}',
                    f'Test Sen:{test_result[-1]:.4f}',
                    f'LR:{self.lr_schedulers[0].lr:.4f}'
                ]))

                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,
                    "Test Loss": self.test_loss.avg,
                    "Test Accuracy": self.test_accuracy.avg,
                    "Val AUC": val_result[0],
                    "Test AUC": test_result[0],
                    'Test Sensitivity': test_result[-1],
                    'Test Specificity': test_result[-2],
                    'micro F1': test_result[-4],
                    'micro recall': test_result[-5],
                    'micro precision': test_result[-6],
                })

                training_process.append({
                    "Epoch": epoch,
                    "Train Loss": self.train_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,
                    "Test Loss": self.test_loss.avg,
                    "Test Accuracy": self.test_accuracy.avg,
                    "Test AUC": test_result[0],
                    'Test Sensitivity': test_result[-1],
                    'Test Specificity': test_result[-2],
                    'micro F1': test_result[-4],
                    'micro recall': test_result[-5],
                    'micro precision': test_result[-6],
                    "Val AUC": val_result[0],
                    "Val Loss": self.val_loss.avg,
                })

            else:

                self.logger.info(" | ".join([
                    f'Epoch[{epoch}/{self.epochs}]',
                    f'Train Loss:{self.train_loss.avg: .3f}',
                    f'Val Loss:{self.val_loss.avg: .3f}',
                    f'Train Accuracy:{self.train_accuracy.avg: .3f}%',
                    f'Test Loss:{self.test_loss.avg: .3f}',
                    f'Test Accuracy:{self.test_accuracy.avg: .3f}%',
                    f'LR:{self.lr_schedulers[0].lr:.4f}'
                ]))

                wandb.log({
                    "Train Loss": self.train_loss.avg,
                    "Val Loss": self.val_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,
                    "Test Loss": self.test_loss.avg,
                    "Test Accuracy": self.test_accuracy.avg,
                    # 'Top 1 Acc': test_result[-3],
                    # 'Top 3 Acc': test_result[-2],
                    # 'Top 5 Acc': test_result[-1],
                    'macro recall': test_result[1],
                    'macro precision': test_result[0],
                })

                training_process.append({
                    "Epoch": epoch,
                    "Train Loss": self.train_loss.avg,
                    "Train Accuracy": self.train_accuracy.avg,
                    "Val Loss": self.val_loss.avg,
                    "Test Loss": self.test_loss.avg,
                    "Test Accuracy": self.test_accuracy.avg,
                    "Val Loss": self.val_loss.avg,
                })

        if self.save_learnable_graph:
            self.generate_save_learnable_matrix()
        self.save_result(training_process)
    
    # change to your own path
    # def graph_heatMap(self, attention):
    #     node_name = None
    #     if self.config.dataset.name == 'abcd':
    #         node = pd.read_csv('ABCD_HCP2016_Node_Information_withAAC6.csv')
    #         node_name = node['AAc-6']
    #     elif self.config.dataset.name == 'pnc':
    #         with open("PNC_v10.txt", 'r') as f:
    #             lines = f.readlines()
    #             node_name = [l[:-1] for l in lines]
    #             node_name = pd.DataFrame(data=node_name, columns=['node'])
    #             node_name = node_name['node']
    #     else:
    #         node = pd.read_csv('ABCD_HCP2016_Node_Information_withAAC6.csv')
    #         node_name = node['AAc-6']

    #     index_list = node_name.unique()

    #     idx = []
    #     divide_line_pos = [0]

    #     region_counter = {}


    #     for group_name in index_list:
    #         tmp = [i for i, r in enumerate(node_name) if r == group_name]
    #         idx += tmp
    #         region_counter[group_name] = 2*len(tmp)*node_name.shape[0]
    #         divide_line_pos.append(len(tmp)+divide_line_pos[-1])

    #     attention = attention[:, idx]
    #     attention = attention[idx, :]

    #     plt.figure(dpi=300)
    #     fig, ax = plt.subplots()
    #     fig.set_size_inches(10, 10)

    #     im = ax.imshow(attention, cmap='coolwarm', vmax=1, vmin=0)

    #     for pos in divide_line_pos[1:-1]:
    #         ax.axvline(x=pos, color='k', linestyle=':', linewidth=1)
    #         ax.axhline(y=pos, color='k', linestyle=':', linewidth=1)

    #     label_pos = []

    #     for i in range(1, len(divide_line_pos)):

    #         label_pos.append((divide_line_pos[i]+divide_line_pos[i-1])/2)

    #     label_pos[-1] += 1

    #     ax.set_xticks([])
    #     ax.set_yticks(label_pos)

    #     ax.set_yticklabels(index_list)
    #     fig.colorbar(im)
    #     plt.tight_layout()
    #     #fig.savefig("test.png")
    #     return fig
    #     #wandb.log({"heat map": fig})
        
    def graph_linePlot(self, data):
        mean_list = np.mean(data, axis = 1)
        std_list = np.std(data, axis = 1)
        up_std = (mean_list + std_list).tolist()
        down_std = (mean_list - std_list).tolist()
        mean_list = mean_list.tolist()
        x_axis = range(len(data))
        
        plt.figure(dpi=300)
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        ax.plot(x_axis, mean_list)
        
        ax.fill_between(x_axis, down_std, up_std, color="grey", alpha = 0.4)
        
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()
        plt.tight_layout()
        #fig.savefig("test.png")
        return fig
    
    def staginWarpInit(self, data, dyn_v):
        dyn_a, sampling_points = self.process_dynamic_fc(data, 50, 30, 600)
        sampling_endpoints = [p+50 for p in sampling_points]
        if dyn_v is None: dyn_v = repeat(torch.eye(360), 'n1 n2 -> b t n1 n2', t=len(sampling_points), b=self.config.dataset.batch_size)
        if len(dyn_a) < self.config.dataset.batch_size: dyn_v = dyn_v[:len(dyn_a)]
        return dyn_a, sampling_endpoints, dyn_v
    
    def process_dynamic_fc(self, timeseries, window_size, window_stride, dynamic_length=None, sampling_init=None, self_loop=True):
        # assumes input shape [minibatch x time x node]
        # output shape [minibatch x time x node x node]
        if dynamic_length is None:
            dynamic_length = timeseries.shape[1]
            sampling_init = 0
        else:
            if isinstance(sampling_init, int):
                assert timeseries.shape[1] > sampling_init + dynamic_length
        assert sampling_init is None or isinstance(sampling_init, int)
        assert timeseries.ndim==3
        assert dynamic_length > window_size

        if sampling_init is None:
            sampling_init = randrange(timeseries.shape[1]-dynamic_length+1)
        sampling_points = list(range(sampling_init, sampling_init+dynamic_length-window_size, window_stride))

        dynamic_fc_list = []
        for i in sampling_points:
            fc_list = []
            for _t in timeseries:
                fc = self.corrcoef(_t[i:i+window_size].T)
                if not self_loop: fc -= torch.eye(fc.shape[0])
                fc_list.append(fc)
            dynamic_fc_list.append(torch.stack(fc_list))
        return torch.stack(dynamic_fc_list, dim=1), sampling_points


    # corrcoef based on
    # https://github.com/pytorch/pytorch/issues/1254
    def corrcoef(self, x):
        mean_x = torch.mean(x, 1, keepdim=True)
        xm = x.sub(mean_x.expand_as(x))
        c = xm.mm(xm.t())
        c = c / (x.size(1) - 1)
        d = torch.diag(c)
        stddev = torch.pow(d, 0.5)
        c = c.div(stddev.expand_as(c))
        c = c.div(stddev.expand_as(c).t())
        c = torch.clamp(c, -1.0, 1.0)
        return c
        
        
