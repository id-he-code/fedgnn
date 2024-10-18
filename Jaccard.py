import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from deeprobust.graph import utils
from deeprobust.graph.defense import GCN_ogbn, GAT_ogbn, GIN_ogbn,GCN

from tqdm import tqdm
import scipy.sparse as sp
import numpy as np
import scipy




class GCNJaccard(GCN):

    def __init__(self, nfeat, nhid, nclass, nnodes=0, binary_feature=True, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, device='cpu'):

        super(GCNJaccard, self).__init__(nfeat, nhid, nclass, dropout, lr, weight_decay, with_relu, with_bias, device=device)
        self.device = device
        self.binary_feature = binary_feature

    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None,
            threshold=0.01, train_iters=200, initialize=True, verbose=True, attention=None):
        print('runing Jaccard')
        self.threshold = threshold
        modified_adj = self.drop_dissimilar_edges(features, adj)
        # modified_adj_tensor = utils.sparse_mx_to_torch_sparse_tensor(self.modified_adj)
        features, modified_adj, labels = utils.to_tensor(features, modified_adj, labels, device=self.device)
        self.modified_adj = modified_adj
        self.features = features
        self.labels = labels
        super().fit(features, modified_adj, labels, idx_train, idx_val, idx_test=None, train_iters=train_iters, initialize=initialize, verbose=verbose)

    def drop_dissimilar_edges(self, features, adj):
        if not sp.issparse(adj):
            adj = sp.csr_matrix(adj)
        modified_adj = adj.copy().tolil()
        # preprocessing based on features

        print('=== GCN-Jaccrad ===')
        # isSparse = sp.issparse(features)
        edges = np.array(modified_adj.nonzero()).T
        removed_cnt = 0
        for edge in tqdm(edges, disable=True): # use disable to disable the progress bar
            n1 = edge[0]
            n2 = edge[1]
            if n1 > n2:
                continue

            # if isSparse:
            if self.binary_feature:
                J = self._jaccard_similarity(features[n1], features[n2])

                if J < self.threshold:
                    modified_adj[n1, n2] = 0
                    modified_adj[n2, n1] = 0
                    removed_cnt += 1
            else:
                # For not binary feature, use cosine similarity
                C = self._cosine_similarity(features[n1], features[n2])
                if C < self.threshold:
                    modified_adj[n1, n2] = 0
                    modified_adj[n2, n1] = 0
                    removed_cnt += 1
        print('removed', removed_cnt ,'edges in the original graph')
        return modified_adj

    def _jaccard_similarity(self, a, b):
        try:
            intersection = a.multiply(b).count_nonzero()
            J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
        except:
            J=0
        return J

    def _cosine_similarity(self, a, b):
        # inner_product = (a * b).sum()
        # C = inner_product / np.sqrt(np.square(a).sum() + np.square(b).sum())
        #Xiang:
        inner_product = a*b.T
        C = inner_product / np.sqrt(a*a.T + b*b.T)
        return C


