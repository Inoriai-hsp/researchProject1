import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.models import (
    FairseqEncoder
)
from fairseq.modules import (
    LayerNorm,
)

from ..modules import init_graphormer_params, GraphormerGraphEncoder

logger = logging.getLogger(__name__)


class PDGGraphormerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # self.max_nodes = args.max_nodes

        self.graph_encoder = GraphormerGraphEncoder(
            # < for graphormer
            num_atoms=84,
            num_in_degree=512,
            num_out_degree=512,
            num_edges=512 * 3,
            num_spatial=512,
            num_edge_dis=128,
            edge_type="multi_hop",
            multi_hop_max_dist=5,
            # >
            num_encoder_layers=6,
            embedding_dim=768,
            ffn_embedding_dim=768,
            num_attention_heads=8,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            encoder_normalize_before=True,
            pre_layernorm=False,
            apply_graphormer_init=True,
            activation_fn="gelu",
        )

        self.share_input_output_embed = False
        self.embed_out = None
        self.lm_output_learned_bias = None

        # Remove head is set to true during fine-tuning
        self.load_softmax = False

        self.masked_lm_pooler = nn.Linear(
            768, 768
        )

        self.lm_head_transform_weight = nn.Linear(
            768, 768
        )
        self.activation_fn = utils.get_activation_fn("gelu")
        self.layer_norm = LayerNorm(768)

        self.lm_output_learned_bias = None
        if self.load_softmax:
            self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))

            if not self.share_input_output_embed:
                self.embed_out = nn.Linear(
                    768, 83, bias=False
                )
            else:
                raise NotImplementedError

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
            self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x

    # def max_nodes(self):
    #     """Maximum output length supported by the encoder."""
    #     return self.max_nodes

    # def upgrade_state_dict_named(self, state_dict, name):
    #     if not self.load_softmax:
    #         for k in list(state_dict.keys()):
    #             if "embed_out.weight" in k or "lm_output_learned_bias" in k:
    #                 del state_dict[k]
    #     return state_dict