import numpy as np
from scipy import stats

from omegaconf import DictConfig
import torch


class preprocess():
    def __init__(self, config):
        self.config = config
        
    def forward(self, dataloader):
        data_all = None
        count = 0
        for datas in dataloader:
            for time_series, _, _ in datas:
                full_sequences = time_series.cpu().detach().numpy()
                for i in range(full_sequences.shape[0]):
                    full_sequence = full_sequences[i, :, :]
                    z_sequence = stats.zscore(full_sequence, axis=1)
                    if data_all is None:
                        data_all = z_sequence
                    else:
                        data_all = np.concatenate((data_all, z_sequence), axis=1)
                    count += 1
                    print(count)
        
        tmp = np.array(data_all)
        np.save('data_all_cnslab_fmri.npy', tmp)

        n_regions = 360
        A = np.zeros((n_regions, n_regions))
        for i in range(n_regions):
            for j in range(i, n_regions):
                if i==j:
                    A[i][j] = 1
                else:
                    A[i][j] = abs(np.corrcoef(data_all[i,:], data_all[j,:])[0][1]) # get value from corrcoef matrix
                    A[j][i] = A[i][j]

        np.save('adj_matrix_cnslab_fmri.npy', A)
        print("Preprocessing done")
