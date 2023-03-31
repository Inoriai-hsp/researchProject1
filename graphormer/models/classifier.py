import torch.nn as nn
from dgl.nn import GraphConv, GlobalAttentionPooling

class ClassifyModel(nn.Module):
    def __init__(self, in_dim, out_dim, num_hidden, num_layers, embedding, encoder, pooler, finetune=False):
        super(ClassifyModel, self).__init__()
        self.finetune = finetune
        self.embedding = embedding
        self.encoder = encoder
        self.pooler = pooler
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.gnn_layers = nn.ModuleList()

        if num_layers == 1:
            self.gnn_layers.append(GraphConv(in_dim, out_dim))
        else:
            self.gnn_layers.append(GraphConv(in_dim, num_hidden, norm='both', activation = nn.ReLU()))
        for l in range(1, num_layers - 1):
            self.gnn_layers.append(GraphConv(num_hidden, num_hidden, norm='both', activation = nn.ReLU()))
        self.gnn_layers.append(GraphConv(num_hidden, out_dim))

        self.gate_nn = nn.Linear(out_dim, 1)
        self.gap = GlobalAttentionPooling(self.gate_nn)

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, int(out_dim / 2)),
            nn.BatchNorm1d(int(out_dim / 2)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(out_dim / 2), int(out_dim / 4)),
            nn.BatchNorm1d(int(out_dim / 4)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(out_dim / 4), 2)
        )

    def forward(self, call_graphs, batch_pdgs):
        if self.finetune:
            x = batch_pdgs.ndata['tac_op']
            edeg_type = batch_pdgs.edata['type']
            x = self.embedding(x)
            x = self.encoder(batch_pdgs, x, edeg_type)
            x = self.pooler(batch_pdgs, x)
        else:
            x = call_graphs.ndata['x']
        
        for l in range(self.num_layers):
            x = self.gnn_layers[l](call_graphs, x)
        x = self.gap(call_graphs, x)
        x = self.mlp(x)
        return x