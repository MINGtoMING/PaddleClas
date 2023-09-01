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

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import XavierNormal, Constant, Normal

xavier_normal_ = XavierNormal()
normal_ = Normal
zeros_ = Constant(value=0.)


class GroupFC(nn.Layer):

    def __init__(self, embedding_size, group_num, class_num):
        super(GroupFC, self).__init__()
        self.embedding_size = embedding_size
        self.group_num = group_num
        self.class_num = class_num

        duplicate_factor = np.ceil(class_num / group_num)
        self.weight = self.create_parameter(
            [group_num, embedding_size, duplicate_factor])
        self.bias = self.create_parameter([class_num], is_bias=True)

    def forward(self, x):
        out = paddle.matmul(x.transpose((1, 0, 2)), self.weight)
        out = out.transpose((1, 0, 2)).flatten(1)
        out = out[:, :self.class_num] + self.bias
        return out


class MultiHeadAttentionIdentity(nn.Layer):

    def __init__(self):
        super(MultiHeadAttentionIdentity, self).__init__()

    def forward(self, *inputs, **kwargs):
        return inputs[0]


class MLDecoder(nn.Layer):

    def __init__(self,
                 class_num=80,
                 query_num=80,
                 in_chans=2048,
                 embed_dim=768,
                 depth=1,
                 num_heads=8,
                 mlp_hidden_dim=2048,
                 dropout=0.1):
        super().__init__()
        self.class_num = class_num

        # 0 < query_num <= class_num
        if query_num > class_num:
            query_num = class_num

        # build non-learnable queries
        self.query_embed = self.create_parameter([query_num, embed_dim])
        self.query_embed.stop_gradient = True

        # build input project
        self.input_proj = nn.Linear(in_chans, embed_dim)

        # build TransformerDecoderLayer
        decoder_layer = nn.TransformerDecoderLayer(
            embed_dim, num_heads, mlp_hidden_dim, dropout)

        # apply RemovalSelfAttn to TransformerDecoderLayer
        decoder_layer.self_attn = MultiHeadAttentionIdentity()
        self.decoder = nn.TransformerDecoder(decoder_layer, depth)

        # build group fully connected weight and bias
        self.group_fc = GroupFC(embed_dim, query_num, class_num)

        self._init_weights()

    def _init_weights(self):
        normal_(self.query_embed)
        xavier_normal_(self.group_fc.weight)
        zeros_(self.group_fc.bias)

    def forward(self, input, label=None):
        batch_size = input.shape[0]
        # input proj
        img_embed = input.flatten(2).transpose((0, 2, 1))
        img_embed = F.relu(self.input_proj(img_embed))
        # TransformerDecoderLayer with RemovalSelfAttn
        query_embed = paddle.tile(
            self.query_embed[None], [batch_size, 1, 1])
        out_embed = self.decoder(query_embed, img_embed)
        # group fully connected
        out = self.group_fc(out_embed)
        return out
