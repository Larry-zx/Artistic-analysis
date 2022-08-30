from model.diffpooling.GCN import GcnEncoderGraph
from model.diffpooling.set2set import Set2Set


class GcnSet2SetEncoder(GcnEncoderGraph):
    def __init__(self, class_num, input_dim=512, hidden_dim=648, embedding_dim=512, num_layers=3,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None):
        super(GcnSet2SetEncoder, self).__init__(class_num, input_dim, hidden_dim, embedding_dim,
                                                num_layers, pred_hidden_dims, concat, bn, dropout, args=args)
        self.s2s = Set2Set(self.pred_input_dim, self.pred_input_dim * 2)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        embedding_tensor = self.gcn_forward(x, adj,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        output = self.s2s(embedding_tensor)
        # out, _ = torch.max(embedding_tensor, dim=1)
        pre_attr = self.pre_head(output)
        return pre_attr
