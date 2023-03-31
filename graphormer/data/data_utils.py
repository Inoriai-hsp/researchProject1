import torch
from tqdm import tqdm
import random
import numpy as np
import dgl
from dgl import load_graphs
from dgl import backend as F

def getSubGraphs(pdg):
    nodes = pdg.ndata['tac_op'].shape[0]
    k = int(nodes / 256) + 1
    partition_ids = dgl.metis_partition_assignment(pdg, k)
    partition_ids = F.asnumpy(partition_ids)
    partition_node_ids = np.argsort(partition_ids)
    partition_size = F.zerocopy_from_numpy(
        np.bincount(partition_ids, minlength=k)
    )
    partition_offset = F.zerocopy_from_numpy(
        np.insert(np.cumsum(partition_size), 0, 0)
    )
    partition_node_ids = F.zerocopy_from_numpy(partition_node_ids)
    sg = [ pdg.subgraph(partition_node_ids[partition_offset[i] : partition_offset[i + 1]], relabel_nodes=True) for i in range(k) ]
    return sg

def getGraphs(sample_number=None):
    address_labels = {}
    with open("/home/huangshiping/study/gigahorse-toolchain/address_labels_gigahorse.txt", "r") as f:
        for line in f.readlines():
            [address, label] = line.strip().split("\t")
            address_labels[address] = torch.tensor(int(label), dtype=torch.int64)
    addresses = list(address_labels.keys())
    if sample_number is not None:
        random.seed(1234)
        pretrain_index = random.sample(range(len(addresses)), sample_number)
    else:
        pretrain_index = range(len(addresses))
    pretrain_graphs = []
    graphs = []
    for i in tqdm(range(len(pretrain_index))):
        address = addresses[pretrain_index[i]]
        pdgs, _ = load_graphs("/home/huangshiping/study/gigahorse-toolchain/PDGs/" + address + ".txt")
        graphs.append((address, pdgs, address_labels[address]))
        pretrain_graphs.extend(pdgs)
    
    return pretrain_graphs, graphs

## shiping
def getCallGraphs(model, pooler, graphs, device):
    call_graphs = []
    # labels = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(graphs))):
            (address, pdgs, label) = graphs[i]
            # labels.append(label.unsqueeze(0))
            batch_g = dgl.batch(pdgs).to(device)
            feat = batch_g.ndata["tac_op"]
            out = model.embed(batch_g, feat)
            out = pooler(batch_g, out)
            srcs = []
            dsts = []
            with open("/home/huangshiping/study/gigahorse-toolchain/CallGraphs/" + address + ".txt", "r") as f:
                for line in f.readlines():
                    [src, dst] = line.strip().split("\t")
                    srcs.append(int(src))
                    dsts.append(int(dst))
            src_ids = torch.tensor(srcs, dtype=torch.int64)
            dst_ids = torch.tensor(dsts, dtype=torch.int64)
            call_graph = dgl.graph((src_ids, dst_ids), num_nodes = len(pdgs))
            call_graph.ndata['x'] = out.cpu()
            call_graphs.append([call_graph, label.unsqueeze(0)])
    return call_graphs

def getCallGraphsForFinetune(graphs):
    call_graphs = []
    for i in tqdm(range(len(graphs))):
        (address, pdgs, label) = graphs[i]
        srcs = []
        dsts = []
        with open("/home/huangshiping/study/gigahorse-toolchain/CallGraphs/" + address + ".txt", "r") as f:
            for line in f.readlines():
                [src, dst] = line.strip().split("\t")
                srcs.append(int(src))
                dsts.append(int(dst))
        src_ids = torch.tensor(srcs, dtype=torch.int64)
        dst_ids = torch.tensor(dsts, dtype=torch.int64)
        call_graph = dgl.graph((src_ids, dst_ids), num_nodes = len(pdgs))
        call_graphs.append((address, call_graph, pdgs, label.unsqueeze(0)))
    return call_graphs