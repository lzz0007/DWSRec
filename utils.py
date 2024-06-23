import torch
import numpy as np
import scipy.sparse as sp
import pandas as pd


def get_norm_adj_mat(interaction_matrix, keep_prob=1):
    r"""Get the normalized interaction matrix of users and items.

    Construct the square matrix from the training data and normalize it
    using the laplace matrix.

    .. math::
        A_{hat} = D^{-0.5} \times A \times D^{-0.5}

    Returns:
        Sparse tensor of the normalized interaction matrix.
    """
    # build adj matrix
    # A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
    # inter_M = self.interaction_matrix[1:, 1:]
    inter_M = interaction_matrix[:, 1:]
    inter_M_t = inter_M.transpose()
    A = inter_M_t.dot(inter_M)
    # norm adj matrix
    sumArr = (A > 0).sum(axis=1)
    # add epsilon to avoid divide by zero Warning
    diag = np.array(sumArr.flatten())[0] + 1e-7
    diag = np.power(diag, -0.5)
    D = sp.diags(diag)
    L = D * A * D
    L = sp.coo_matrix(L)

    # # edge type
    # e2t_adj = {}
    # for u, v in zip(*L.nonzero()):
    #     e2t_adj[(int(u), int(v))] = 0

    # covert norm_adj matrix to tensor
    row = L.row
    col = L.col
    i = torch.LongTensor([row, col])
    data = torch.FloatTensor(L.data)
    SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))

    # if keep_prob < 1:
    #     size = L.size()
    #     index = L.indices().t()
    #     values = L.values()
    #     random_index = torch.rand(len(values)) + keep_prob  # gen values from 0 to 1
    #     random_index = random_index.int().bool()
    #     index = index[random_index]
    #     values = values[random_index] / keep_prob
    #     SparseL = torch.sparse.FloatTensor(index.t(), values, size)
    # else:
    #     # covert norm_adj matrix to tensor
    #     row = L.row
    #     col = L.col
    #     i = torch.LongTensor([row, col])
    #     data = torch.FloatTensor(L.data)
    #     SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
    # return SparseL.to_dense(), e2t_adj
    return SparseL


def build_knn_neighbourhood(adj, topk):
    knn_val, knn_ind = torch.topk(adj, topk, dim=-1)
    weighted_adjacency_matrix = (torch.zeros_like(adj)).scatter_(-1, knn_ind, knn_val)
    return weighted_adjacency_matrix


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

    return L_norm


def build_sim(context):
    context_norm = context.div(torch.norm(context, p=2, dim=-1, keepdim=True))
    sim = torch.mm(context_norm, context_norm.transpose(1, 0))
    return sim


def whitening(x, engine, group):
    # code from "On feature decorrelation in self-supervised learning"
    N, D = x.shape
    # G = math.ceil(2 * D / N)
    G = group
    # if dim_shuffle:
    #     if training:
    #         new_idx = torch.randperm(D)
    #     else:
    #         new_idx = torch.arange(D)
    # else:
    #     new_idx = torch.arange(D)
    new_idx = torch.arange(D)
    x = x.t()[new_idx].t()
    x = x.view(N, G, D // G)
    x = (x - x.mean(dim=0, keepdim=True)).transpose(0, 1)  # G, N, D//G
    covs = x.transpose(1, 2).bmm(x) / N
    W = transformation(covs, x.device, engine=engine)
    x = x.bmm(W)
    output = x.transpose(1, 2).flatten(0, 1)[torch.argsort(new_idx)].t()
    # output = self.white_linear(output)
    return output


def transformation(covs, device, engine='symeig'):
    covs = covs.to(device)
    if engine == 'cholesky':
        C = torch.cholesky(covs.to(device))
        W = torch.triangular_solve(torch.eye(C.size(-1)).expand_as(C).to(C), C, upper=False)[0].transpose(1, 2).to(device)
    else:
        if engine == 'symeig':
            S, U = torch.symeig(covs.to(device), eigenvectors=True, upper=True)
        elif engine == 'svd':
            U, S, _ = torch.svd(covs.to(device))
        elif engine == 'svd_lowrank':
            U, S, _ = torch.svd_lowrank(covs.to(device))
        elif engine == 'pca_lowrank':
            U, S, _ = torch.pca_lowrank(covs.to(device), center=False)
        S, U = S.to(device), U.to(device)
        W = U.bmm(S.rsqrt().diag_embed()).bmm(U.transpose(1, 2))
    return W
