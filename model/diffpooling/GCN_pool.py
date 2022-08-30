from model.diffpooling.GCN import GcnEncoderGraph
import torch
import torch.nn as nn


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, class_num, max_num_nodes, input_dim=512, hidden_dim=648, embedding_dim=512,
                 assign_hidden_dim=648, num_layers=2, assign_ratio=0.5, assign_num_layers=-1, num_pooling=2,
                 pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, linkpred=True,
                 assign_input_dim=-1, args=None):
        '''
        Args:
            num_layers: number of gc layers before each pooling
            num_nodes: number of nodes for each graph in batch
            linkpred: flag to turn on link prediction side objective
        '''

        super(SoftPoolingGcnEncoder, self).__init__(class_num, input_dim, hidden_dim, embedding_dim, num_layers,
                                                    pred_hidden_dims=pred_hidden_dims, concat=concat,
                                                    args=args)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        # GC 输出都是embedding_dim
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, num_layers,
                add_self, normalize=True, dropout=dropout)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self,
                normalize=True)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        if 'assign_x' in kwargs:
            x_a = kwargs['assign_x']
        else:
            x_a = x

        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        # out_all = []

        # Z = GCN_{embed}(A,X)
        embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last,
                                            embedding_mask)  # [batch_size x num_nodes x embedding_dim]
        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None
            # S = Softmax(GCN_{pool} (A,X))
            self.assign_tensor = self.gcn_forward(x_a, adj,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)
            # [batch_size x num_nodes x next_lvl_num_nodes]
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            # pool后再接一层
            embedding_tensor = self.gcn_forward(x, adj,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out_all = []
            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.mean(embedding_tensor, dim=1)
                # out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        pre_attr = self.pre_head(output)

        return pre_attr

    def loss(self, adj=None, batch_num_nodes=None, adj_hop=1):
        '''
        Args:
            batch_num_nodes: numpy array of number of nodes in each graph in the minibatch.
        '''
        eps = 1e-7
        max_num_nodes = adj.size()[1]
        pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
        tmp = pred_adj0
        pred_adj = pred_adj0
        for adj_pow in range(adj_hop - 1):
            tmp = tmp @ pred_adj0
            pred_adj = pred_adj + tmp
        pred_adj = torch.min(pred_adj, torch.ones(1, dtype=pred_adj.dtype).to(cfg.device))
        self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
        if batch_num_nodes is None:
            num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
            print('Warning: calculating link pred loss without masking')
        else:
            num_entries = np.sum(batch_num_nodes * batch_num_nodes)
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
            self.link_loss[(1 - adj_mask).bool()] = 0.0

        self.link_loss = torch.sum(self.link_loss) / float(num_entries)
        return self.link_loss
