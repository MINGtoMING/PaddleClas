# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import XavierNormal, Constant, Normal

xavier_normal_ = XavierNormal()
normal_ = Normal
zeros_ = Constant(value=0.)


class ChoiceIdentity(nn.Layer):
    """
    ChoiceIdentity is a variant of nn.Identity,
    which can return specified input.
    """

    def __init__(self, index=None):
        super().__init__()
        self.index = index

    def single_choice(self, *args, **kwargs):
        index = kwargs.pop("index")
        if isinstance(index, int):
            assert 0 <= index < len(args), \
                f"Index {index} will be out of range!!!"
            return args[index]
        elif isinstance(index, str):
            assert index in kwargs, \
                f"Do not include the key called {index}!!!"
            return kwargs[index]
        else:
            raise NotImplementedError(
                f"Currently, this type {type(index)} is not supported!!!")

    def forward(self, *args, **kwargs):
        if isinstance(self.index, (list, tuple)):
            output = []
            for idx in self.index:
                kwargs["index"] = idx
                output.append(self.single_choice(*args, **kwargs))
        else:
            kwargs["index"] = self.index
            return self.single_choice(*args, **kwargs)


class MLDecoder(nn.Layer):
    """
    ML-Decoder is an attention-based classification head,
    which introduced by Tal Ridnik et al. in https://arxiv.org/pdf/2111.12933.pdf.
    """

    def __init__(self,
                 class_num=1000,
                 in_chans=2048,
                 query_num=100,
                 embed_dim=768,
                 depth=1,
                 num_heads=8,
                 mlp_hidden_dim=2048,
                 dropout=0.1,
                 activation="relu",
                 self_attn_removal=True):
        super().__init__()
        self.class_num = class_num
        self.in_chans = in_chans

        query_num = min(max(query_num, 1), class_num)

        self.input_proj = nn.Conv2D(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1)

        self.query_pos_embed = nn.Embedding(
            num_embeddings=query_num,
            embedding_dim=embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_hidden_dim,
            dropout=dropout,
            activation=activation,
            attn_dropout=dropout,
            act_dropout=dropout)
        if self_attn_removal:
            decoder_layer.self_attn = ChoiceIdentity(index=0)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=depth)

        group_factor = math.ceil(class_num / query_num)
        self.group_conv = nn.Conv2D(
            in_channels=query_num * embed_dim,
            out_channels=query_num * group_factor,
            kernel_size=1,
            stride=1,
            groups=query_num)

        self._init_weights()

    def _init_weights(self):
        self.query_pos_embed.weight.stop_gradient = True
        normal_(self.query_pos_embed.weight)
        xavier_normal_(self.group_conv.weight[:, :self.class_num])
        zeros_(self.group_conv.bias)

    def group_fc_pool(self, x):
        x = x.flatten(1)[..., None, None]
        x = self.group_conv(x)
        x = x.flatten(1)[:, :self.class_num]
        return x

    def forward(self, x):
        assert x.ndim == 4 and x.shape[1] == self.in_chans, "Wrong input shape!!!"

        feat_proj = F.relu(self.input_proj(x))
        feat_flatten = feat_proj.flatten(2).transpose([0, 2, 1])

        query_pos_embed = self.query_pos_embed.weight[None].tile([x.shape[0], 1, 1])
        out_embed = self.decoder(query_pos_embed, feat_flatten)

        logit = self.group_fc_pool(out_embed)
        return logit
