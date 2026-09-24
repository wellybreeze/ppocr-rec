# ppocr-rec 使用指南

面向新手的识别全流程：安装 → 数据准备 → 训练 → 验证 → 预测 → 导出与多后端加载。

本仓库基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)，用 PyTorch 实现，许可证为 [Apache 2.0](../LICENSE)。只识别已经裁好的文字行，不做检测。发行名是 `ppocr-rec`，命令行入口是 `ppocr-rec`，Python 入口是 `from ppocr_rec import OCR`。参数写法为 `key=value`。预训练权重在 [Releases v0.0.1](https://github.com/wellybreeze/ppocr-rec/releases/tag/v0.0.1)，不随源码打包。

---

## 1. 安装

在仓库根目录（克隆后为 `ppocr-rec/`）：

```bash
pip install -e .
# 导出 ONNX 并用 ONNX Runtime 预测时再装：
pip install -e ".[export]"
```

需要：Python ≥ 3.9、PyTorch ≥ 2.0、OpenCV。有 NVIDIA GPU 时安装对应的 CUDA 版 PyTorch。

验证安装：

```bash
ppocr-rec predict model=ppocrv5_mobile.yaml source=word.jpg
```

未提供权重时这只是随机初始化网络，结果无意义，但能确认命令可用。

---

## 2. 五分钟走通

下面用包内 CRNN 配方在你自己的小数据集上跑通一遍（图片高度按 CRNN 用 32；PP-OCRv3 及以后用 48，见第 4 节）。

```bash
# 1) 准备好第 3 节的目录和 rec.yaml 后：
ppocr-rec train model=crnn.yaml data=rec.yaml epochs=50 batch=64 device=0

# 2) 验证（自动读 runs/rec/exp/weights/best.pt 也可以显式指定）
ppocr-rec val model=runs/rec/exp/weights/best.pt data=rec.yaml

# 3) 预测一张图或一个文件夹
ppocr-rec predict model=runs/rec/exp/weights/best.pt source=word.jpg
ppocr-rec predict model=runs/rec/exp/weights/best.pt source=crops/

# 4) 导出并加载导出模型再预测
ppocr-rec export model=runs/rec/exp/weights/best.pt format=onnx
ppocr-rec predict model=runs/rec/exp/best.onnx source=word.jpg
```

Python 等价写法：

```python
from ppocr_rec import OCR

model = OCR("crnn.yaml")
model.train(data="rec.yaml", epochs=50, batch=64, device=0)
metrics = model.val(data="rec.yaml")          # {'acc': ..., 'norm_edit_dis': ...}
results = model.predict("word.jpg")
print(results[0].text, results[0].conf)

onnx_path = model.export(format="onnx")
exported = OCR(onnx_path)
print(exported.predict("word.jpg")[0].text)
```

训练结束后 `OCR` 会自动加载 `best.pt`，因此上面的 `val` / `predict` / `export` 可以直接接着用。

---

## 3. 数据准备

识别任务吃的是 **已经裁好的文字行/单词图**，不是整页文档。检测框请先用检测模型裁出后再送进来。

### 3.1 推荐目录结构

标注文件里的路径相对于数据集 yaml 里的 `path`。两种常见摆法：

**方案 A：yaml 和图片根目录在一起（推荐）**

```text
my_rec/
  rec.yaml
  train_list.txt
  val_list.txt
  dict.txt                 # 可选，自定义字典时才需要
  train/
    000001.jpg
    000002.jpg
    ...
  val/
    000001.jpg
    ...
```

对应 `rec.yaml`：

```yaml
path: .                    # 相对 rec.yaml 所在目录
train: train_list.txt
val: val_list.txt
dict: ppocrv5_dict.txt     # 或 dict.txt
delimiter: "\t"
use_space_char: true
```

**方案 B：yaml 在外面，数据在子目录**

```text
my_rec/
  rec.yaml
  train_data/
    train_list.txt
    val_list.txt
    train/...
    val/...
```

```yaml
path: ./train_data         # 相对 rec.yaml 所在目录解析
train: train_list.txt
val: val_list.txt
dict: ppocrv5_dict.txt
delimiter: "\t"
use_space_char: true
```

包内完整注释模板：`ppocr_rec/cfg/datasets/rec.example.yaml`。

`path` 也可写成绝对路径，例如 `D:/data/rec`。

### 3.2 标注文件

- 编码：**UTF-8**
- 一行一个样本
- 默认分隔符是 **Tab**（`\t`），不是空格
- 格式：`相对path的图片路径<TAB>标签文本`

`train_list.txt` 示例：

```text
train/000001.jpg	你好世界
train/000002.jpg	Hello
train/subdir/000003.png	OCR-2024
train/000004.jpg	空 格 也 可以
```

注意：

- 图片路径不要写成 `path` 的绝对路径再拼一层。`path` 已经是根，列表里写相对根的路径。
- 标签里可以再出现 Tab（只按第一个 Tab 切开路径和文本）。
- 空行会被跳过；缺分隔符的行会被跳过。
- 读不到的图片会变成黑图且标签被置空，等于脏样本，训练前请确认路径。
- 支持后缀：`.jpg` `.jpeg` `.png` `.bmp` `.webp` `.tif` `.tiff`（预测扫目录时同样如此）。

若必须用空格当分隔符（不推荐，路径里不能有空格）：

```yaml
delimiter: " "
```

### 3.3 字符字典

字典决定「能识别哪些字」。**标签里出现字典没有的字符，该样本在 CTC 编码时会被丢弃（标签变空）**，等于白训。

格式：UTF-8，**每行一个字符**，不要表头。

```text
0
1
2
a
b
你
好
```

`use_space_char: true` 时，若文件里还没有空格，会自动把空格追加进字典（英文、混排建议打开）。

包内自带字典（写 yaml 的 `dict:` 时用文件名即可，不必拷贝）：

| 文件 | 适用 |
|------|------|
| `ppocr_keys_v1.txt` | 通用中英（CRNN / SVTR / PP-OCRv2/v3/v4） |
| `ppocrv5_dict.txt` | PP-OCRv5 |
| `ppocrv6_tiny_dict.txt` | PP-OCRv6 tiny |
| `ppocrv6_dict.txt` | PP-OCRv6 small / medium |

自定义场景（只认数字、只认车牌等）请自己做一份小字典，能明显减小分类头、加快收敛。

微调官方预训练权重时，**字典必须与预训练一致**（或能被权重加载逻辑兼容）。分类层宽度对不上的参数会被自动跳过，等于那一层随机初始化。

### 3.4 文本长度 `max_text_length`

默认 **25** 个字符。超过该长度的样本不会被正确编码（CTC 标签为空）。

- 标签普遍更长：训练时加大 `max_text_length=40`，并同步改模型 yaml 里 NRTR/SAR 的 `max_text_length`（若该模型有 MultiHead）。
- 更稳妥的做法是在数据侧把过长行切成多张图。

### 3.5 图像尺寸习惯

预处理是：保持高宽比缩放到高度 `img_h`，宽度不足则右侧重填，超过则限制到 `img_w`。像素归一化到 `[-1, 1]`。

| 配方 | 默认 `imgsz` `[高, 宽]` |
|------|-------------------------|
| `crnn.yaml` / `svtr.yaml` | `[32, 100]` |
| `svtr_ch.yaml` / `ppocrv2.yaml` | `[32, 320]` |
| PP-OCRv3 / v4 / v5 / v6 | `[48, 320]` |

`svtr.yaml`（英文 SVTR）会把图直接拉到固定 `32×100`（`padding: false`），其它配方默认按宽高比填充。

现在训练 / 验证 / 预测 / 导出若未显式传 `imgsz`，会自动用模型 yaml 里的值。**不要用 32 高的配方硬训 48，也不要反过来。**

### 3.6 训练集规模建议

| 场景 | 大致量级 |
|------|----------|
| 数字/短英文，从头训 | 几千～几万 |
| 通用中文，从头训 | 十万级更稳 |
| 有官方预训练，微调自己的字体/场景 | 几百～几万即可 |

验证集建议单独划出 5%～10%，不要和训练列表重复。

---

## 4. 选哪个模型

YAML 都在 `ppocr_rec/cfg/models/`。`OCR("ppocrv5_mobile.yaml")` 或 `OCR("ppocrv5_mobile")` 都可以。

| YAML | 骨干 | 训练头 | 增强 | 适用 |
|------|------|--------|------|------|
| `crnn.yaml` | MobileNetV3 | CTC | RecAug | 轻量入门、英文数字 |
| `ppocrv2.yaml` | MobileNetV1Enhance | CTC | RecAug | 中英通用偏老配方 |
| `svtr.yaml` | SVTRNet 32×100 | CTC | SVTR RecAug | 英文场景 |
| `svtr_ch.yaml` | SVTRNet 32×320 | CTC | RecConAug + RecAug | 中文 SVTR |
| `ppocrv3_mobile.yaml` | MobileNetV1Enhance | CTC + SAR | RecConAug + RecAug | 移动端 |
| `ppocrv4_mobile.yaml` / `ppocrv5_mobile.yaml` | PPLCNetV3 | CTC + NRTR | RecConAug + RecAug + 多尺度 | **默认推荐（手机/CPU）** |
| `ppocrv4_server.yaml` | PPHGNet_small | CTC + NRTR | RecConAug + RecAug + 多尺度 | 服务端 |
| `ppocrv5_server.yaml` | PPHGNetV2_B4 | CTC + NRTR | RecAug + 多尺度 | 服务端高精度 |
| `ppocrv6_tiny.yaml` | PPLCNetV4 tiny | CTC + NRTR | RecConAug + RecAug + 多尺度 | 极小模型 |
| `ppocrv6_small.yaml` / `ppocrv6_medium.yaml` | PPLCNetV4 | CTC + LightSVTR + NRTR | RecConAug + RecAug + 多尺度 | v6 精度档 |

验证 / 预测 / 导出时 MultiHead 只走 **CTC 分支**（和 PaddleOCR 推理一致）。PP-OCRv3 及以后识别池化按高度 **48** 设计。

已发布的预训练权重、官方精度和下载链接见 [README 的预训练权重](../README.md#预训练权重)。下载后把路径传给 `model=` 即可预测，不必再转换：

```bash
wget -O ppocrv5_mobile.pt \
  https://github.com/wellybreeze/ppocr-rec/releases/download/v0.0.1/ppocrv5_mobile.pt
ppocr-rec predict model=ppocrv5_mobile.pt source=word.jpg
```

未迁移：ParseQ、ABINet、SATRN、SRN 以及带 TPS/STN 的 STARNet/RARE 等。

---

## 5. 训练

### 5.1 命令

```bash
ppocr-rec train model=ppocrv5_mobile.yaml data=rec.yaml epochs=100 batch=64 device=0
```

```python
from ppocr_rec import OCR

model = OCR("ppocrv5_mobile.yaml")
model.train(
    data="rec.yaml",
    epochs=100,
    batch=64,
    device=0,
    pretrained="pretrained/ppocrv5_mobile.pt",  # 可选
    project="runs/rec",
    name="v5_ft",
)
```

命令行列表参数要带括号，例如 `imgsz=[48,320]`。

### 5.2 输出目录

```text
runs/rec/exp/                 # name 默认 exp，重名会变成 exp2、exp3
  args.yaml                   # 本次全部超参
  results.csv
  weights/
    last.pt                   # 每个 epoch 覆盖保存（断点续训读这个）
    best.pt                   # 验证集 acc 最好的一份
    epoch10.pt                # 仅当 save_period>0 时，每 N 个 epoch 额外留一份
```

指定目录：`project=runs/rec name=exp1 exist_ok=true`（`exist_ok=true` 时允许覆盖同名实验）。

断点续训会写入**同一个**实验目录，不会另开 `exp2`。

### 5.3 超参数含义与怎么调

未写出的项来自 `ppocr_rec/cfg/default.yaml`，命令行 / Python 关键字会覆盖它们。

| 参数 | 默认 | 含义 | 怎么调 |
|------|------|------|--------|
| `data` | 无 | 数据集 yaml 路径 | 必填 |
| `model` | — | 结构 yaml 或 `.pt` | 见第 4 节 |
| `epochs` | 100 | 训练轮数 | 小数据 50～100，看 val `acc` 是否还在涨；大数据可 150～300 |
| `batch` | 64 | 批大小 | 显存不够就减半（32/16/8）。过小（<8）时把 `lr0` 也略降 |
| `imgsz` | 跟模型 yaml | `[高, 宽]` | 先与配方一致。文本普遍很长可只加大宽度（如 `[48, 640]`）；有 `pos_embed` 的 SVTR 不要随便改空间尺寸 |
| `device` | 自动 | `0` / `cuda:0` / `cpu` | 多卡本版本按单卡用，指定一张即可 |
| `workers` | 4 | DataLoader 进程数 | Linux 4～8；Windows / 部分 WSL 卡住时改 `0` |
| `optimizer` | Adam | `Adam` / `AdamW` / `SGD` | 默认 Adam。AdamW 适合想把权重衰减做干净的情况；SGD 需更小心调 `lr0` 和 `momentum` |
| `lr0` | 0.0005 | 初始学习率 | **从头训练**保持 5e-4；**微调预训练**建议 1e-4～5e-5。loss 炸了就降 10 倍 |
| `lrf` | 0.0 | 最终学习率 = `lr0 * lrf` | 余弦退火末尾。`0` 表示收到 0 附近；想留一点尾部学习率可设 `0.01` |
| `cos_lr` | true | 余弦退火 | 一般保持 true |
| `warmup_epochs` | 5 | 前几轮线性升温 | 数据很少可降到 1～2；很大可保持 5 |
| `weight_decay` | 3e-5 | 权重衰减 | 过拟合可略增（1e-4）；欠拟合减到 0 |
| `momentum` | 0.937 | 仅 SGD | 用 SGD 时再管它 |
| `amp` | true | CUDA 混合精度 | GPU 上保持 true；CPU 会自动不走 AMP。出现 NaN 可 `amp=false` |
| `pretrained` | 无 | 预训练 `.pt` | 官方 Paddle 权重需先转换（第 9 节）。字典宽度不一致的分类层会跳过 |
| `seed` | 0 | 随机种子 | 复现实验时固定 |
| `val` | true | 每个 epoch 后验证 | 极大数据集想加速可 `val=false`，训完再单独 `val` |
| `verbose` | true | 进度条 | 日志重定向时可关掉 |
| `project` / `name` | `runs/rec` / `exp` | 实验目录 | 每次改 `name` 以免覆盖 |
| `exist_ok` | false | 是否复用同名目录 | 调试时 true；`resume` 指向已有 run 时会自动当作 true |
| `max_text_length` | 25 | 标签最大字符数 | 见 3.4 |
| `save` | true | 是否写权重 | 关掉则不写 `last.pt` / `best.pt` / `epochN.pt` |
| `patience` | 50 | 验证 `acc` 连续多少 epoch 不提升就停 | 小数据可 10～20；想跑满 `epochs` 设 `patience=0` 关闭早停 |
| `resume` | false | 从 `last.pt` 接着训 | 见下方示例。会恢复权重、优化器、学习率、已完成 epoch 和 best acc |
| `fraction` | 1.0 | 训练集抽样比例 | `0.1` 只拿 10% 训练样本做通路/调参，验证集仍用全量。与 `seed` 一起可复现子集 |
| `save_period` | -1 | 每隔 N 个 epoch 另存 `epochN.pt` | `-1` 关闭。例如 `save_period=10` 得到 `epoch10.pt`、`epoch20.pt` |

每个 epoch 仍会更新 `last.pt`；验证 `acc` 创新高时更新 `best.pt`。

**早停**只看验证集。`val=false` 时不会触发。`patience=0` 表示不早停。

**断点续训**（中断后把 `epochs` 设成最终目标轮数，不要改成「还剩多少」）：

```bash
# 方式 1：直接指向 last.pt（推荐）
ppocr-rec train model=ppocrv5_mobile.yaml data=rec.yaml epochs=100 \
  resume=runs/rec/exp/weights/last.pt

# 方式 2：resume=true 时到 project/name/weights/last.pt 找文件
ppocr-rec train model=ppocrv5_mobile.yaml data=rec.yaml epochs=100 \
  resume=true project=runs/rec name=exp

# 方式 3：从 last.pt 构造 OCR 再 resume=true
```

```python
from ppocr_rec import OCR

OCR("runs/rec/exp/weights/last.pt").train(data="rec.yaml", epochs=100, resume=True)
```

**抽样调参**：`fraction=0.1 seed=0` 先在子集上试学习率，确认后再 `fraction=1.0` 全量训。

### 5.4 增强

配方（开不开拼接、用 RecAug 还是 SVTRRecAug、多尺度）仍由模型 yaml 的 `aug:` 和算法名决定。各操作的**概率**可在 CLI / `default.yaml` 里改，**默认等于 Paddle 原设定**：

| 参数 | 默认 | 含义 |
|------|------|------|
| `rec_aug` | `true` | 关闭则不做 RecAug / SVTRRecAug（不影响 `con_aug`） |
| `tia_prob` | 0.4 | TIA 扭曲/拉伸/透视 |
| `crop_prob` | 0.4 | 上下轻裁 |
| `blur_prob` | 0.4 | 高斯模糊 |
| `hsv_aug_prob` | 0.4 | Paddle RecAug 的微弱 V 通道抖动 |
| `jitter_prob` | 0.4 | 像素平移 jitter |
| `noise_prob` | 0.4 | 高斯噪声 |
| `reverse_prob` | 0.4 | 反色 |
| `gray_prob` | 0.0 | 灰度化后再变回 3 通道。Paddle 没有此项，默认关 |
| `bgr2rgb_prob` | 0.0 | 训练时随机 BGR↔RGB。Paddle 没有此项，默认关 |
| `con_aug` | 跟配方 | `true`/`false` 覆盖模型 yaml；不填则 MultiHead 默认开、v5 server / 英文 SVTR 关 |
| `con_aug_num` | 跟配方（一般 2） | 拼接时额外抽几张 |
| `con_aug_prob` | 0.5 | RecConAug 拼接概率 |
| `multi_scale` | 跟配方 | 覆盖 v4–v6 的 `[[320,32],[320,48],[320,64]]` |
| `geometry_p` | 0.5 | 仅 SVTRRecAug |
| `deterioration_p` | 0.25 | 仅 SVTRRecAug |
| `colorjitter_p` | 0.25 | 仅 SVTRRecAug |

```bash
# 关掉几何 TIA，打开一点灰度化：
ppocr-rec train model=ppocrv5_mobile.yaml data=rec.yaml tia_prob=0 gray_prob=0.1
# 整套 RecAug 关掉，只保留拼词：
ppocr-rec train model=ppocrv5_mobile.yaml data=rec.yaml rec_aug=false
# 关掉拼接（也可写在模型 yaml 的 aug.con_aug）：
ppocr-rec train model=ppocrv5_mobile.yaml data=rec.yaml con_aug=false
```

```yaml
# 模型 yaml 里关拼接
aug:
  con_aug: false
```

### 5.5 微调官方权重

```bash
ppocr-rec train \
  model=ppocrv5_mobile.yaml \
  data=rec.yaml \
  pretrained=/path/to/ppocrv5_mobile.pt \
  epochs=50 \
  lr0=0.0001 \
  batch=64
```

字典与官方一致时分类层会接上；你换成更小的自定义字典时，头层会重新训，骨干仍能当特征用。

### 5.6 怎么判断有没有训好

看验证日志里的：

- `acc`：整串完全匹配准确率（比较时忽略空格）
- `norm_edit_dis`：归一化编辑距离，越接近 1 越好

`acc` 很高但线上仍错：检查字典是否缺字、验证集是否和训练同分布、输入高度是否为 48（v3+）。训练 `acc` 很高、验证很低：减小增强或加数据，略增 `weight_decay`。

---

## 6. 验证

```bash
ppocr-rec val model=runs/rec/exp/weights/best.pt data=rec.yaml batch=64 device=0
```

```python
from ppocr_rec import OCR

model = OCR("runs/rec/exp/weights/best.pt")
metrics = model.val(data="rec.yaml", split="val")
print(metrics["acc"], metrics["norm_edit_dis"])
```

- `split=val` 默认用 yaml 的 `val:` 列表；可改成 `split=train` 做训练集过拟合检查。
- 验证不做数据增强，尺寸与 `imgsz` 一致。
- 也可对导出后的 TorchScript / ONNX 做验证（仍需要 `data=` 指向带标签的 yaml）。

---

## 7. 预测

```bash
ppocr-rec predict model=best.pt source=word.jpg
ppocr-rec predict model=best.pt source=crops/ batch=16
ppocr-rec predict model=best.pt source=word.jpg save_txt=true
```

```python
from ppocr_rec import OCR

model = OCR("best.pt")
for r in model.predict("crops/"):
    print(r.path, r.text, r.conf)
```

`source` 可以是：单张图、目录（递归）、路径列表、或 BGR `numpy.ndarray`。

`save_txt=true` 时在本次 `project/name` 下写入 `labels/<stem>.txt`，内容为 `文本<TAB>置信度`。

同一 batch 会按宽度分组并 pad；带 `pos_embed` 的 SVTR 会强制用满 `imgsz` 宽度。

---

## 8. 导出与多后端加载

从 **PyTorch `.pt` 或结构 yaml** 导出（不要对已经导出的文件再 export）。

### 8.1 支持的一等格式

| `format` | 产物 | 加载预测 | 依赖 |
|----------|------|----------|------|
| `torchscript`（默认，别名 `ts` / `pt` / `jit`） | `*.torchscript` | `OCR("xxx.torchscript")` | 仅 PyTorch |
| `onnx` | `*.onnx` | `OCR("xxx.onnx")` | `pip install ppocr-rec[export]`（onnx + onnxruntime + onnxsim） |

ONNX 默认会：

1. **图内归一化** `(x - mean) / std`（默认 mean/std=0.5）→ 推理输入只需 `/255`，右侧 pad 填 mean
2. **图内 CTC 后处理** → 输出 `text_ids [B, max_text_length] int32`（无效位 `-1`）和 `text_confs [B, 1]`
3. 把字符表写入 ONNX `character` metadata（没有旁边的 `dict.txt` 也能解码）

```bash
ppocr-rec export model=best.pt format=torchscript
ppocr-rec export model=best.pt format=onnx opset=17 dynamic=true
ppocr-rec export model=best.pt format=onnx \
  normalize=true mean=[0.5,0.5,0.5] std=[0.5,0.5,0.5] \
  postprocess=true ignored_tokens=[0] remove_duplicate=true \
  max_text_length=32 simplify=true
# 只要 softmax、不要图内解码/归一化：
ppocr-rec export model=best.pt format=onnx postprocess=false normalize=false
# 指定输出路径，并用一张图做导出后校验：
ppocr-rec export model=best.pt format=onnx output=deploy/rec.onnx test_image=word.jpg
```

```python
from ppocr_rec import OCR

model = OCR("best.pt")
ts = model.export(format="torchscript", imgsz=[48, 320])
onnx = model.export(
    format="onnx",
    imgsz=[48, 320],
    dynamic=True,
    opset=17,
    normalize=True,
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5],
    postprocess=True,
    ignored_tokens=[0],
    remove_duplicate=True,
    max_text_length=32,
    simplify=True,
    output="deploy/rec.onnx",
    test_image="word.jpg",
)
raw = model.export(format="onnx", postprocess=False, normalize=False)
```

相关参数：

| 参数 | 默认 | 含义 |
|------|------|------|
| `format` | `torchscript` | `torchscript` 或 `onnx` |
| `model` | — | 输入权重 `.pt` 或结构 yaml |
| `output` / `output_model` | 自动 | 导出文件完整路径；不填则写到 `project/name/` |
| `imgsz` | 跟模型 / 权重 | 导出时的示意输入高宽 |
| `dynamic` | true | ONNX 动态 batch 与 **宽度**（高度固定）。变长文本请保持 true |
| `opset` | 17 | ONNX opset |
| `device` | 自动 | 跟踪/导出所用设备 |
| `normalize` | true | ONNX 图内 `(x-mean)/std`。关掉则输入需已是 `[-1,1]` |
| `mean` / `std` | `[0.5, 0.5, 0.5]` | 图内归一化均值/方差，需与训练预处理一致 |
| `postprocess` | true | ONNX 内嵌 CTC greedy |
| `ignored_tokens` | `[0]` | 后处理忽略的字符 id（CTC blank 为 0） |
| `remove_duplicate` | true | 去掉连续重复字符。`false` 则保留 |
| `max_text_length` | 25 | `text_ids` 宽度；有效字符多于此时会被截断 |
| `simplify` | false | 用 onnxsim 简化图；未安装则跳过并告警 |
| `test_image` | 空 | 导出后用该图跑一次预测并打印结果 |

导出目录示例：

```text
runs/rec/exp/
  best.onnx              # 或 best.torchscript
  dict.txt               # 字符表（ONNX 里也有一份 metadata）
  metadata.yaml          # format / imgsz / normalize / postprocess / 字典名
```

**`*.onnx` + `metadata.yaml` 建议一起保留。** 端到端 ONNX 即使没有 `dict.txt`，只要图里带了 `character` metadata 也能 `OCR("best.onnx")` 预测。

若字典被挪走了且 metadata 也没有：

```python
OCR("best.onnx", character_dict="my_dict.txt")
```

```bash
ppocr-rec predict model=best.onnx source=word.jpg character_dict=my_dict.txt
```

### 8.2 导出后加载预测

```python
from ppocr_rec import OCR

# TorchScript
m = OCR("runs/rec/exp/best.torchscript")
print(m.predict("word.jpg")[0].text)

# ONNX Runtime（默认端到端：图内归一化 + CTC；GPU 时优先 CUDAExecutionProvider）
m = OCR("runs/rec/exp/best.onnx")
print(m.predict("word.jpg")[0].text)
```

```bash
ppocr-rec predict model=runs/rec/exp/best.torchscript source=word.jpg
ppocr-rec predict model=runs/rec/exp/best.onnx source=word.jpg
```

接口与 `.pt` 相同：仍返回 `RecResult(text, conf, path)`。含 LSTM 的配方（如 `crnn.yaml`）导出的 ONNX，预测时建议 `batch=1`。

### 8.3 其它推理引擎

端到端 ONNX 可直接给 TensorRT / OpenVINO / 独立 ONNX Runtime：

- 输入名 `images`，`N×3×H×W` float32，数值为 **`/255` 后的 [0,1]**（`normalize=true` 时）；右侧不足宽度填 `0.5`
- 输出 `text_ids`、`text_confs`；`text_ids[i] >= 0` 的位置按字典下标拼字（0 是 blank，已被滤掉）

`postprocess=false` 时输出仍是 softmax `output`，需自己做 CTC greedy。

`default.yaml` 里的 `int8` / `keras` / `nms` 尚未实现。

---

## 9. 从 PaddleOCR 转预训练

脚本在仓库内：`tools/convert_paddle_rec.py`。在仓库根目录执行。这一步单独安装 Paddle 用来读 `.pdparams`，训练和预测仍然不依赖它。

```bash
pip install paddlepaddle
python tools/convert_paddle_rec.py \
  --pdparams path/to/best_accuracy.pdparams \
  --yaml ppocrv5_mobile.yaml \
  --out pretrained/ppocrv5_mobile.pt \
  --dict ppocr_rec/cfg/dicts/ppocrv5_dict.txt
```

| 参数 | 含义 |
|------|------|
| `--pdparams` | PaddleOCR 训练得到的 `best_accuracy.pdparams` |
| `--yaml` | 与该权重大小对应的模型配方，如 `ppocrv5_mobile.yaml` |
| `--out` | 输出的 PyTorch checkpoint |
| `--dict` | 该权重使用的字符字典；官方中英模型用包内 `ppocr_rec/cfg/dicts/` 下同名文件 |
| `--strict-report` | 打印未能映射或未被使用的参数名 |

得到的 `.pt` 作为 `pretrained=` 微调，或 `OCR("pretrained/ppocrv5_mobile.pt")` 直接预测。分类层宽度与新字典不一致时会自动跳过，骨干网络权重仍然加载。蒸馏 checkpoint 若同时含 Student / Teacher，只保留 Student。

---

## 10. 常见问题

**Q: 中文预测全是乱码 / 空串？**  
字典和权重不匹配，或还在用随机初始化的 yaml。请加载 `best.pt` / 转换后的预训练，并确认 `dict` 一致。

**Q: 训练很快但 acc 一直是 0？**  
检查标注分隔符是不是 Tab、图片路径是否相对 `path`、字典是否覆盖标签字符、`imgsz` 高度是否正确。

**Q: OOM？**  
减小 `batch`；v5/v6 server 先用 `batch=8/16`；关掉多尺度可改模型 yaml 去掉 `multi_scale`（精度会变）。

**Q: Windows 里 DataLoader 报错？**  
`workers=0`。

**Q: 想识别超过 25 个字的长行？**  
加大 `max_text_length` 与输入宽度，或先把行切短。仅加宽 `imgsz` 而不改 `max_text_length` 不够。

**Q: CLI 里 `imgsz=[48,320]` 被当成字符串？**  
请升级到带列表解析的版本；必须写成带方括号的 Python 字面量，不要写成 `48,320`。

---

## 特别鸣谢

本仓库基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的识别配方与数据处理，采用 PyTorch 实现，许可证为 [Apache 2.0](../LICENSE)。接口组织参考了 [Ultralytics](https://github.com/ultralytics/ultralytics) 官方库。感谢两个项目的开源贡献。
