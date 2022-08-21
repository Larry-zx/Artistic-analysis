import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GAT.layers import GraphAttentionLayer, SpGraphAttentionLayer
from model.base_network import BaseNetwork
import config as cfg


class GAT(BaseNetwork):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class SpGAT(BaseNetwork):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat,
                                                 nhid,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads,
                                             nclass,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GAT_multi(BaseNetwork):
    def __init__(self, nfeat=512, nhid=128, dropout=0.5, alpha=0.2, nheads=6):
        """Dense version of GAT."""
        super(GAT_multi, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att_artist = GraphAttentionLayer(nhid * nheads, 129, dropout=dropout, alpha=alpha, concat=False)
        self.out_att_genre = GraphAttentionLayer(nhid * nheads, 11, dropout=dropout, alpha=alpha, concat=False)
        self.out_att_style = GraphAttentionLayer(nhid * nheads, 27, dropout=dropout, alpha=alpha, concat=False)

        self.head_artist = nn.Linear(cfg.k_node + 1, 1)
        self.head_genre = nn.Linear(cfg.k_node + 1, 1)
        self.head_style = nn.Linear(cfg.k_node + 1, 1)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        F_artist = F.elu(self.out_att_artist(x, adj)).permute(0, 2, 1)
        F_genre = F.elu(self.out_att_genre(x, adj)).permute(0, 2, 1)
        F_style = F.elu(self.out_att_style(x, adj)).permute(0, 2, 1)
        pre_artist = self.head_artist(F_artist).squeeze(-1)
        pre_genre = self.head_genre(F_genre).squeeze(-1)
        pre_style = self.head_style(F_style).squeeze(-1)
        return {'artist': pre_artist, 'genre': pre_genre, 'style': pre_style}
