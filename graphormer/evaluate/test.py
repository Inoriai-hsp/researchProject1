from graphmae.datasets.data_utils import getGraphs
from graphmae.datasets.dgl_dataset import GraphormerDataset, BatchedDataDataset, TargetDataset, EpochShuffleDataset
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset,
)
import numpy as np
import torch

from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf

pretrain_graphs, graphs = getGraphs(sample_number=10)

dataset = GraphormerDataset(pretrain_graphs)
dataset_val = dataset.dataset_val
batched_data = BatchedDataDataset(dataset_val)
print(len(batched_data))
data_sizes = np.array([batched_data.max_node] * len(batched_data))

target = TargetDataset(batched_data)

dataset = NestedDictionaryDataset(
    {
        "nsamples": NumSamplesDataset(),
        "net_input": {"batched_data": batched_data},
        "target": target,
    },
    sizes=data_sizes,
)

dataset = EpochShuffleDataset(
    dataset, num_samples=len(dataset), seed=0
)
task = tasks.setup_task("graph_prediction")

batch_iterator = task.get_batch_iterator(
    dataset=dataset,
    max_tokens=dataset.max_tokens_valid,
    max_sentences=dataset.batch_size_valid,
    max_positions=utils.resolve_max_positions(
        task.max_positions(),
        # model.max_positions(),
    ),
    ignore_invalid_inputs=dataset.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=dataset.required_batch_size_multiple,
    seed=0,
    num_workers=dataset.num_workers,
    epoch=0,
    data_buffer_size=dataset.data_buffer_size,
    disable_iterator_cache=False,
)
itr = batch_iterator.next_epoch_itr(
    shuffle=False, set_dataset_epoch=False
)
# progress = progress_bar.progress_bar(
#     itr,
#     log_format=cfg.common.log_format,
#     log_interval=cfg.common.log_interval,
#     default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple")
# )

# # infer
# y_pred = []
# y_true = []
# with torch.no_grad():
#     model.eval()
#     for i, sample in enumerate(progress):
#         sample = utils.move_to_cuda(sample)
#         y = model(**sample["net_input"])[:, 0, :].reshape(-1)
#         y_pred.extend(y.detach().cpu())
#         y_true.extend(sample["target"].detach().cpu().reshape(-1)[:y.shape[0]])
#         torch.cuda.empty_cache()

# # save predictions
# y_pred = torch.Tensor(y_pred)
# y_true = torch.Tensor(y_true)

