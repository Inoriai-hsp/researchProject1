import torch
import numpy as np
from tqdm import tqdm
from dgl.data import DGLDataset

from .wrapper import convert_to_single_emb
from . import algos

class PDGDataset(DGLDataset):
    def __init__(self, dataset, url=None, raw_dir=None, save_dir=None, hash_key=(), force_reload=False, verbose=False, transform=None):
        # super(PDGDataset, self).__init__("PDGDataset", url, raw_dir, save_dir, hash_key, force_reload, verbose, transform)
        self.dataset = dataset
        # self.dataset = []
        self.__process()

    def __process(self):
        for idx in tqdm(range(len(self.dataset))):
            graph_data = self.dataset[idx]
            N = graph_data.num_nodes()

            node_int_feature = graph_data.ndata['tac_op'].unsqueeze(1)
            edge_int_feature = graph_data.edata['type'].unsqueeze(1)
            edge_index = graph_data.edges()
            attn_edge_type = torch.zeros(
                [N, N, edge_int_feature.shape[1]], dtype=torch.long
            )
            attn_edge_type[
                edge_index[0].long(), edge_index[1].long()
            ] = convert_to_single_emb(edge_int_feature)
            dense_adj = graph_data.adj().to_dense().type(torch.int)
            shortest_path_result, path = algos.floyd_warshall(dense_adj.numpy())
            max_dist = np.amax(shortest_path_result)
            edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
            spatial_pos = torch.from_numpy((shortest_path_result)).long()
            attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

            graph_data.x = convert_to_single_emb(node_int_feature)
            graph_data.adj = dense_adj
            graph_data.attn_bias = attn_bias
            graph_data.attn_edge_type = attn_edge_type
            graph_data.spatial_pos = spatial_pos
            graph_data.in_degree = dense_adj.long().sum(dim=1).view(-1)
            graph_data.out_degree = graph_data.in_degree
            graph_data.edge_input = torch.from_numpy(edge_input).long()
            graph_data.idx = idx
            graph_data.y = torch.tensor(1).unsqueeze(-1)

            # self.dataset.append(graph_data)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)