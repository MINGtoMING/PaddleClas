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
from paddle.io import Dataset
from pycocotools.coco import COCO

from ppcls.data.preprocess import transform
from ppcls.utils import logger
from .common_dataset import create_operators


class MultiLabelCOCODataset(Dataset):
    """
    Load multi-label dataset with COCO format.

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        transform_ops (list, optional): list of transform op(s). Defaults to None.
    """

    def __init__(self,
                 dataset_dir,
                 image_dir,
                 anno_path,
                 transform_ops=None):
        super().__init__()
        image_dir = os.path.join(dataset_dir, image_dir)
        anno_path = os.path.join(dataset_dir, anno_path)
        assert os.path.exists(image_dir) and os.path.exists(anno_path), \
            ValueError("The dataset is not Found or "
                       "the folder structure is non-conformance.")

        self._transform_ops = create_operators(transform_ops)

        self.images = []
        self.labels = []

        coco = COCO(anno_path)
        cat_ids = coco.getCatIds()
        catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})

        assert 'annotations' in coco.dataset, \
            'Annotation file: {} does not contains ground truth!!!'.format(anno_path)

        for img_id in sorted(list(coco.imgToAnns.keys())):
            img_info = coco.loadImgs([img_id])[0]
            img_filename = img_info['file_name']
            img_w = img_info['width']
            img_h = img_info['height']

            img_filepath = os.path.join(image_dir, img_filename)
            if not os.path.exists(img_filepath):
                logger.warning('Illegal image file: {}, '
                               'and it will be ignored'.format(img_filepath))
                continue

            self.images.append(img_filepath)

            if img_w < 0 or img_h < 0:
                logger.warning(
                    'Illegal width: {} or height: {} in annotation, '
                    'and im_id: {} will be ignored'.format(img_w, img_h, img_id))
                continue

            ins_anno_ids = coco.getAnnIds(imgIds=[img_id])
            instances = coco.loadAnns(ins_anno_ids)

            label = np.zeros([len(cat_ids)], dtype=np.uint8)
            for instance in instances:
                label[catid2clsid[instance['category_id']]] = 1

            self.labels.append(label)

        self.class_num = len(cat_ids)

    def __getitem__(self, idx):
        with open(self.images[idx], 'rb') as f:
            img = f.read()

        if self._transform_ops:
            img = transform(img, self._transform_ops)

        img = img.transpose([2, 0, 1])

        return img, self.labels[idx]

    def __len__(self):
        return len(self.images)
