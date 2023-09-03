# ML-Decoder 系列

<a name='1'></a>

## 1. 模型介绍

<a name='1.1'></a>

### 1.1 模型简介

待补充。

<a name='1.2'></a>

### 1.2 主干模型在ImageNet1K上的指标

|        Models        | Top1  | FLOPs<br>(G) | Params<br>(M) |
|:--------------------:|:-----:|:------------:|:-------------:|
|       ResNet50       | 0.765 |    8.190     |    25.560     |
|      ResNet50*       | 0.804 |    8.190     |    25.560     |
|     ResNet50_vd      | 0.791 |    8.670     |    25.580     |
|   ResNet50_vd_ssld   | 0.830 |    8.670     |    25.580     |
| Fix_ResNet50_vd_ssld | 0.840 |    17.696    |    25.580     |

**备注：** `*` 表示使用timm库所提供的预训练权重 。

### 1.3 主干模型结合ML-Decoder在COCO2017多分类任务上的指标

|        Models        | use_ml_decoder | mAP(%) | FLOPs<br>(G) | Params<br>(M) |
|:--------------------:|:--------------:|:------:|:------------:|:-------------:|
|       ResNet50       |      False     |   -    |      -       |       -       |
|       ResNet50       |      True      |  78.4  |      -       |       -       |
|      ResNet50*       |      False     |   -    |      -       |       -       |
|      ResNet50*       |      True      |   -    |      -       |       -       |
|     ResNet50_vd      |      False     |   -    |      -       |       -       |
|     ResNet50_vd      |      True      |   -    |      -       |       -       |
|   ResNet50_vd_ssld   |      False     |   -    |      -       |       -       |
|   ResNet50_vd_ssld   |      True      |   -    |      -       |       -       |
| Fix_ResNet50_vd_ssld |      False     |   -    |      -       |       -       |
| Fix_ResNet50_vd_ssld |      True      |   -    |      -       |       -       |

**备注：** `*` 表示使用timm库所提供的预训练权重 。

## 3. 模型训练、评估和预测

### 3.1 数据准备

请在[MSCOCO 官网](https://cocodataset.org/)准备 ImageNet-1k 相关的数据。

进入 PaddleClas 目录。

```
cd path_to_PaddleClas
```

进入 `dataset/` 目录，将下载好的数据命名为 `COCO2017` ，存放于此。 `COCO2017` 目录中具有以下数据：

```
├── train2017
├── val2017
├── annotations
│   ├── instances_train2017.json
│   ├── instances_val2017.json
│   ├── ...
```

其中 `train2017/` 和 `val2017/` 分别为训练集和验证集。`instances_train2017.json` 和 `instances_val2017.json` 分别为训练集和验证集的标签文件。

<a name="3.3"></a>

### 3.3 模型训练

<a name="3.3.1"></a>

#### 3.3.1 训练 ImageNet

在 `ppcls/configs/ImageNet/ResNet/ResNet50.yaml` 中提供了 ResNet50 训练配置，可以通过如下脚本启动训练：

```shell
export CUDA_VISIBLE_DEVICES=0,1,2,3
python3 -m paddle.distributed.launch \
    --gpus="0,1,2,3" \
    tools/train.py \
        -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml
```

**备注：**

* 当前精度最佳的模型会保存在 `output/ResNet50/best_model.pdparams`

<a name="3.3.2"></a>

#### 3.3.2 基于 ImageNet 权重微调

如果训练的不是 ImageNet 任务，而是其他任务时，需要更改配置文件和训练方法，详情可以参考：[模型微调](../../training/single_label_classification/finetune.md)。

<a name="3.4"></a>

### 3.4 模型评估

训练好模型之后，可以通过以下命令实现对模型指标的评估。

```bash
python3 tools/eval.py \
    -c ppcls/configs/ImageNet/ResNet/ResNet50.yaml \
    -o Global.pretrained_model=output/ResNet50/best_model
```

其中 `-o Global.pretrained_model="output/ResNet50/best_model"` 指定了当前最佳权重所在的路径，如果指定其他权重，只需替换对应的路径即可。
