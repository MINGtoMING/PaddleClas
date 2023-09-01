#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import os

import numpy as np
from pycocotools.coco import COCO

from ppcls.data.dataloader.common_dataset import CommonDataset


class COCODataset(CommonDataset):
    """COCODataset for multi-label classification task in COCO.

    Args:
        image_root (str): image root, path to `COCO2014`
        cls_label_path (str): path to annotation file `instances_train2014.json` or `instances_val2014.json`
        transform_ops (list, optional): list of transform op(s). Defaults to None.
        delimiter (str, optional): delimiter. Defaults to None.
        relabel (bool, optional): whether you do relabel when original label do not start from 0 or are discontinuous. Defaults to False.
    """

    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 delimiter=None,
                 relabel=False):
        self.delimiter = delimiter if delimiter is not None else " "
        self.relabel = relabel
        super(COCODataset, self).__init__(image_root, cls_label_path,
                                          transform_ops)

    def _load_anno(self):
        assert os.path.exists(self._img_root)
        assert os.path.exists(self._cls_path)
        self.images = []
        self.labels = []

        coco = COCO(self._cls_path)
        cat_ids = coco.getCatIds()
        catid2clsid = {catid: i for i, catid in enumerate(cat_ids)}
        for img_id in list(sorted(coco.getImgIds())):
            img_fname = coco.loadImgs([img_id])[0]['file_name']
            img_filepath = os.path.join(self._img_root, img_fname)
            assert os.path.exists(img_filepath)
            self.images.append(img_filepath)

            ins_anno_ids = coco.getAnnIds(imgIds=[img_id])
            instances = coco.loadAnns(ins_anno_ids)
            gt_label = np.zeros((len(cat_ids), ), dtype=np.float32)
            for inst in instances:
                gt_label[catid2clsid[inst['category_id']]] = 1
            self.labels.append(gt_label)
