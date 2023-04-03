from graphormer.data.data_utils import getGraphs
from graphormer.data.pdg_dataset import PDGDataset
from dgl.dataloading import GraphDataLoader
from graphormer.data.collator import collator
from graphormer.models.pdg_graphormer import PDGGraphormerEncoder
from fairseq import utils

dataset, _ = getGraphs(10)
dataset = PDGDataset(dataset)
# for data in dataset:
#     print(data)

dataloader = GraphDataLoader(dataset, collator, batch_size = 64, shuffle=True)
model = PDGGraphormerEncoder()
model.to(device=1)
for batch_data in dataloader:
    batch_data = utils.move_to_cuda(batch_data, device=1)
    output = model(batch_data)
