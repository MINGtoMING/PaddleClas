# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class MultiLabelASL(nn.Layer):
    """
    Multi-label asymmetric loss
    """

    def __init__(self,
                 gamma_pos=1,
                 gamma_neg=4,
                 clip=0.05,
                 epsilon=1e-8,
                 disable_focal_loss_grad=True,
                 reduction="sum"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.epsilon = epsilon
        self.disable_focal_loss_grad = disable_focal_loss_grad
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

    def forward(self, x, target):
        if isinstance(x, dict):
            x = x["logits"]
        pred_sigmoid = F.sigmoid(x)

        # Asymmetric Clipping and Basic CE calculation
        if self.clip and self.clip > 0:
            pt = (1 - pred_sigmoid + self.clip).clip(max=1) \
                 * (1 - target) + pred_sigmoid * target
        else:
            pt = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target

        # Asymmetric Focusing
        if self.disable_focal_loss_grad:
            paddle.set_grad_enabled(False)
        asymmetric_weight = (1 - pt).pow(
            self.gamma_pos * target + self.gamma_neg * (1 - target))
        if self.disable_focal_loss_grad:
            paddle.set_grad_enabled(True)

        loss = -paddle.log(pt.clip(min=self.epsilon)) * asymmetric_weight
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return {"MultiLabelASL": loss}
