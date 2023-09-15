import torch
import numpy as np
from sklearn.neighbors import KernelDensity


def discrete_mixup_data(x, y, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    mix_m = torch.bernoulli(torch.ones(*x.shape[1:])*lam).to(device)

    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = mix_m * x + (1 - mix_m) * x[index, :]

    y = lam * y + (1-lam) * y[index]
    return mixed_x, y


def continus_mixup_data(*xs, y=None, alpha=1.0, device='cpu'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    new_xs = [lam * x + (1 - lam) * x[index, :] for x in xs]
    y = lam * y + (1-lam) * y[index]
    return *new_xs, y


def mixup_renn_data(x, log_nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_nodes = []
    for i, j in enumerate(index):
        mixed_nodes.append(torch.matrix_exp(
            lam * log_nodes[i] + (1 - lam) * log_nodes[j]))

    mixed_nodes = torch.stack(mixed_nodes)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, mixed_nodes, y_a, y_b, lam


def mixup_data_by_class(x, nodes, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''

    mix_xs, mix_nodes, mix_ys = [], [], []

    for t_y in y.unique():
        idx = y == t_y

        t_mixed_x, t_mixed_nodes, _, _, _ = continus_mixup_data(
            x[idx], nodes[idx], y[idx], alpha=alpha, device=device)
        mix_xs.append(t_mixed_x)
        mix_nodes.append(t_mixed_nodes)

        mix_ys.append(y[idx])

    return torch.cat(mix_xs, dim=0), torch.cat(mix_nodes, dim=0), torch.cat(mix_ys, dim=0)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def cal_step_connect(connectity, step):
    multi_step = connectity
    for _ in range(step):
        multi_step = np.dot(multi_step, connectity)
    multi_step[multi_step > 0] = 1
    return multi_step


def obtain_partition(dataloader, fc_threshold, step=2):
    pearsons = []
    for data_in, pearson, label in dataloader:
        pearsons.append(pearson)

    fc_data = torch.mean(torch.cat(pearsons), dim=0)

    fc_data[fc_data > fc_threshold] = 1
    fc_data[fc_data <= fc_threshold] = 0

    _, n = fc_data.shape

    final_partition = torch.zeros((n, (n-1)*n//2))

    connection = cal_step_connect(fc_data, step)
    temp = 0
    for i in range(connection.shape[0]):
        temp += i
        for j in range(i):
            if connection[i, j] > 0:
                final_partition[i, temp-i+j] = 1
                final_partition[j, temp-i+j] = 1

    connect_num = torch.sum(final_partition > 0)/n
    print(f'Final Partition {connect_num}')

    return final_partition.float()


def mixup_cluster_loss(matrixs, y, intra_weight=2):

    y_1 = y[:, 1]

    y_0 = y[:, 0]

    bz, roi_num, _ = matrixs.shape
    matrixs = matrixs.reshape((bz, -1))
    sum_1 = torch.sum(y_1)
    sum_0 = torch.sum(y_0)
    loss = 0.0

    if sum_0 > 0:
        center_0 = torch.matmul(y_0, matrixs)/sum_0
        diff_0 = torch.norm(matrixs-center_0, p=1, dim=1)
        loss += torch.matmul(y_0, diff_0)/(sum_0*roi_num*roi_num)
    if sum_1 > 0:
        center_1 = torch.matmul(y_1, matrixs)/sum_1
        diff_1 = torch.norm(matrixs-center_1, p=1, dim=1)
        loss += torch.matmul(y_1, diff_1)/(sum_1*roi_num*roi_num)
    if sum_0 > 0 and sum_1 > 0:
        loss += intra_weight * \
            (1 - torch.norm(center_0-center_1, p=1)/(roi_num*roi_num))

    return loss


def inner_loss(label, matrixs):

    loss = 0

    if torch.sum(label == 0) > 1:
        loss += torch.mean(torch.var(matrixs[label == 0], dim=0))

    if torch.sum(label == 1) > 1:
        loss += torch.mean(torch.var(matrixs[label == 1], dim=0))

    return loss


def intra_loss(label, matrixs):
    a, b = None, None

    if torch.sum(label == 0) > 0:
        a = torch.mean(matrixs[label == 0], dim=0)

    if torch.sum(label == 1) > 0:
        b = torch.mean(matrixs[label == 1], dim=0)
    if a is not None and b is not None:
        return 1 - torch.mean(torch.pow(a-b, 2))
    else:
        return 0


def tensor_log(t):
    # t, u, v = 16, 264, 264, s = 16, 264
    # u, s, v = torch.svd(t)
    s, u = torch.linalg.eigh(t)
    s[s <= 0] = 1e-8
    return u @ torch.diag_embed(torch.log(s)) @ u.permute(0, 2, 1)


def tensor_exp(t):
    # condition: t is symmetric!
    s, u = torch.linalg.eigh(t)
    return u @ torch.diag_embed(torch.exp(s)) @ u.permute(0, 2, 1)


def log_euclidean_distance(A, B):
    inner_term = tensor_log(A) - tensor_log(B)
    inner_multi = inner_term @ inner_term.permute(0, 2, 1)
    _, s, _ = torch.svd(inner_multi)
    final = torch.sum(s, dim=-1)
    return final


def renn_mixup(x, y, alpha=1.0, device='cuda'):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    # lam = 1
    # x += torch.diag(torch.ones(x.size()[1])) * 1e-6
    batch_size = y.size()[0]
    index = torch.randperm(batch_size).to(device)
    x = tensor_log(x)
    x = lam * x + (1 - lam) * x[index, :]
    y = lam * y + (1-lam) * y[index]
    return tensor_exp(x), y


def get_mixup_sample_rate(data_packet, device='cuda', use_kde=True, kde_type='gaussian', kde_bandwidth=1.0):

    mix_idx = []
    _, y_list = data_packet['x_train'], data_packet['y_train']

    data_list = y_list

    N = len(data_list)

    ######## use kde rate or uniform rate #######
    for i in range(N):
        if use_kde:  # kde
            data_i = data_list[i]
            data_i = data_i.reshape(-1, data_i.shape[0])  # get 2D

            ######### get kde sample rate ##########
            kd = KernelDensity(kernel=kde_type, bandwidth=kde_bandwidth).fit(
                data_i)  # should be 2D
            each_rate = np.exp(kd.score_samples(data_list))
            each_rate[i] = 0
            each_rate /= np.sum(each_rate)  # norm
        else:
            each_rate = np.ones(y_list.shape[0]) * 1.0 / y_list.shape[0]

        mix_idx.append(each_rate)

    mix_idx = np.array(mix_idx)

    return mix_idx


def get_batch_kde_mixup_idx(Batch_X, Batch_Y, device):
    if Batch_X.shape[0] % 2 != 0:
        return list(range(Batch_X.shape[0]))
    Batch_packet = {}
    Batch_packet['x_train'] = Batch_X.cpu()
    Batch_packet['y_train'] = Batch_Y.cpu()

    Batch_rate = get_mixup_sample_rate(
        Batch_packet, device, use_kde=True)  # batch -> kde

    idx2 = [np.random.choice(np.arange(Batch_X.shape[0]), p=Batch_rate[sel_idx])
            for sel_idx in np.arange(Batch_X.shape[0])]
    return idx2


def get_batch_kde_mixup_batch(x, y, alpha=1.0, device='cuda'):

    index = get_batch_kde_mixup_idx(x, y, device)

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    x = lam * x + (1 - lam) * x[index, :]
    y = lam * y + (1-lam) * y[index]
    return x, y


def drop_edge(x, drop_prob=0.2):

    sample_num, node_sz = x.shape[0], x.shape[1]
    index = np.triu_indices(node_sz, k=1)
    x = x[:, index[0], index[1]]

    mask = torch.rand(x.shape) < drop_prob

    x[mask] = 0

    new_sample = torch.ones((sample_num, node_sz, node_sz))
    new_sample[:, index[0], index[1]] = x
    new_sample[:, index[1], index[0]] = x
    return new_sample


def drop_node(x, drop_prob=0.2):

    node_sz = x.shape[1]

    mask = torch.rand(node_sz) < drop_prob

    x[:, mask, :] = 0
    x[:, :, mask] = 0

    return x
