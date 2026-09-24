# ppocr-rec

本仓库基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 识别配方，用 **PyTorch** 实现训练、验证、预测与导出（TorchScript / ONNX），导出后可直接加载再预测。运行时不依赖 Paddle。只识别已经裁好的文字行，不做检测。

预训练权重在 [GitHub Releases](https://github.com/wellybreeze/ppocr-rec/releases/tag/v0.0.1)，不随源码打包。直接加载 yaml 是随机初始化。

发行名 `ppocr-rec`，Python 包 `ppocr_rec`，命令行 `ppocr-rec`。许可证为 [Apache 2.0](LICENSE)。

结构对齐 PaddleOCR（CRNN、SVTR、PP-OCRv3/v4/v5/v6）：Transform → Backbone → Neck → Head。

**完整新手教程（数据目录、标注、超参、导出加载）：** [docs/usage.md](docs/usage.md)

## 安装

需要 Python ≥ 3.9、PyTorch ≥ 2.0。有 NVIDIA GPU 时安装对应的 CUDA 版 PyTorch。

```bash
cd ppocr-rec
pip install -e .
pip install -e ".[export]"    # ONNX 导出 + ONNX Runtime 预测
```

## 快速上手

```python
from ppocr_rec import OCR

model = OCR("ppocrv5_mobile.yaml")
model.train(data="path/to/rec.yaml", epochs=75)
metrics = model.val()
results = model.predict("word.jpg")
onnx_path = model.export(format="onnx")
print(OCR(onnx_path).predict("word.jpg")[0].text)
```

```bash
ppocr-rec train model=ppocrv5_mobile.yaml data=rec.yaml epochs=75
ppocr-rec val model=best.pt data=rec.yaml
ppocr-rec predict model=best.pt source=word.jpg
ppocr-rec export model=best.pt format=onnx
ppocr-rec predict model=best.onnx source=word.jpg
```

数据集 yaml（标注为 `相对路径<TAB>文本`，完整说明见 [docs/usage.md](docs/usage.md#3-数据准备)）：

```yaml
path: ./train_data
train: train_list.txt
val: val_list.txt
dict: ppocrv5_dict.txt
delimiter: "\t"
use_space_char: true
```

模板：`ppocr_rec/cfg/datasets/rec.example.yaml`。

## 模型

YAML 位于 `ppocr_rec/cfg/models/`：

| YAML | Backbone | Head | 训练增强 |
|------|----------|------|----------|
| `crnn.yaml` | MobileNetV3 | CTC | RecAug |
| `ppocrv2.yaml` | MobileNetV1Enhance | CTC | RecAug |
| `svtr.yaml` | SVTRNet 32×100 | CTC | SVTRRecAug |
| `svtr_ch.yaml` | SVTRNet 32×320 | CTC | RecConAug + RecAug |
| `ppocrv3_mobile.yaml` | MobileNetV1Enhance | CTC + SAR | RecConAug + RecAug |
| `ppocrv4_mobile.yaml` / `ppocrv5_mobile.yaml` | PPLCNetV3 | CTC + NRTR | RecConAug + RecAug + 多尺度 |
| `ppocrv4_server.yaml` | PPHGNet_small | CTC + NRTR | RecConAug + RecAug + 多尺度 |
| `ppocrv5_server.yaml` | PPHGNetV2_B4 | CTC + NRTR | RecAug + 多尺度 |
| `ppocrv6_tiny.yaml` | PPLCNetV4 tiny | CTC + NRTR | RecConAug + RecAug + 多尺度 |
| `ppocrv6_small.yaml` / `ppocrv6_medium.yaml` | PPLCNetV4 | CTC (LightSVTR) + NRTR | RecConAug + RecAug + 多尺度 |

多语言 YAML（latin/korean/…）与蒸馏/AMP 变体共用上表结构，换字典即可。未迁移：ParseQ、ABINet、SATRN、SRN、RFL、PREN、CAN、VisionLAN、UniMERNet、公式识别、带 TPS/STN 的 STARNet/RARE 等学术模型。

RecAug 对齐 Paddle：TIA + crop/blur/HSV/jitter/高斯噪声/反色，概率默认同 Paddle（`tia_prob` 等可在训练参数里改）。v4–v6 多尺度为 `[[320,32],[320,48],[320,64]]`。

MultiHead 在 val / predict / export 只走 CTC。PP-OCRv3/v4/v5/v6 识别池化期望输入高度 **48**。

## 预训练权重

下列 `.pt` 由官方 Paddle 训练权重转换而来，可直接预测或作为 `pretrained=` 微调。精度和耗时引自 [PaddleOCR 文本识别模块](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/module_usage/text_recognition.md)，是官方模型在其评估集上的结果，不是本仓库重测。PP-OCRv6 带 `*` 的精度来自内部多场景评估集，与 v3–v5 的通用评估集不同，不能直接对比。耗时只含 Paddle 推理，格式为「常规 / 高性能」，单位毫秒。

| 模型 | YAML | 字典 | Avg Accuracy (%) | GPU (ms) | CPU (ms) | 下载 |
|------|------|------|------------------|----------|----------|------|
| `ppocrv6_medium` | `ppocrv6_medium.yaml` | `ppocrv6_dict.txt` | 83.2* | — | — | [ppocrv6_medium.pt](https://github.com/wellybreeze/ppocr-rec/releases/download/v0.0.1/ppocrv6_medium.pt) |
| `ppocrv6_small` | `ppocrv6_small.yaml` | `ppocrv6_dict.txt` | 81.3* | — | — | [ppocrv6_small.pt](https://github.com/wellybreeze/ppocr-rec/releases/download/v0.0.1/ppocrv6_small.pt) |
| `ppocrv6_tiny` | `ppocrv6_tiny.yaml` | `ppocrv6_tiny_dict.txt` | 73.5* | — | — | [ppocrv6_tiny.pt](https://github.com/wellybreeze/ppocr-rec/releases/download/v0.0.1/ppocrv6_tiny.pt) |
| `ppocrv5_server` | `ppocrv5_server.yaml` | `ppocrv5_dict.txt` | 86.38 | 8.46 / 2.36 | 31.21 / 31.21 | [ppocrv5_server.pt](https://github.com/wellybreeze/ppocr-rec/releases/download/v0.0.1/ppocrv5_server.pt) |
| `ppocrv5_mobile` | `ppocrv5_mobile.yaml` | `ppocrv5_dict.txt` | 81.29 | 5.43 / 1.46 | 21.20 / 5.32 | [ppocrv5_mobile.pt](https://github.com/wellybreeze/ppocr-rec/releases/download/v0.0.1/ppocrv5_mobile.pt) |
| `ppocrv4_server` | `ppocrv4_server.yaml` | `ppocr_keys_v1.txt` | 85.19 | 8.75 / 2.49 | 36.93 / 36.93 | [ppocrv4_server.pt](https://github.com/wellybreeze/ppocr-rec/releases/download/v0.0.1/ppocrv4_server.pt) |
| `ppocrv4_mobile` | `ppocrv4_mobile.yaml` | `ppocr_keys_v1.txt` | 78.74 | 5.26 / 1.12 | 17.48 / 3.61 | [ppocrv4_mobile.pt](https://github.com/wellybreeze/ppocr-rec/releases/download/v0.0.1/ppocrv4_mobile.pt) |
| `ppocrv3_mobile` | `ppocrv3_mobile.yaml` | `ppocr_keys_v1.txt` | 72.96 | 3.89 / 1.16 | 8.72 / 3.56 | [ppocrv3_mobile.pt](https://github.com/wellybreeze/ppocr-rec/releases/download/v0.0.1/ppocrv3_mobile.pt) |

字典在 `ppocr_rec/cfg/dicts/`。checkpoint 里已保存配套字典，加载 `.pt` 后可直接预测：

```bash
wget -O ppocrv5_mobile.pt \
  https://github.com/wellybreeze/ppocr-rec/releases/download/v0.0.1/ppocrv5_mobile.pt
ppocr-rec predict model=ppocrv5_mobile.pt source=word.jpg
```

```python
from ppocr_rec import OCR

model = OCR("ppocrv5_mobile.pt")
print(model.predict("word.jpg")[0].text)
```

## 导出

| format | 文件 | 再预测 |
|--------|------|--------|
| `torchscript`（默认） | `*.torchscript` | `OCR("a.torchscript")` |
| `onnx` | `*.onnx` | `OCR("a.onnx")`（需 `[export]` 额外依赖） |

ONNX 默认图内归一化 + CTC 后处理（`text_ids` / `text_confs`）。可用 `normalize` / `postprocess` / `ignored_tokens` / `remove_duplicate` / `mean` / `std` / `opset` / `simplify` / `output` / `test_image` 控制。详见 [docs/usage.md](docs/usage.md#8-导出与多后端加载)。

## Paddle 权重转换

在仓库根目录执行。训练和预测不需要 Paddle；只有这一步要先 `pip install paddlepaddle` 才能读取 `.pdparams`。`--dict` 用与该权重配套的字典，包内字典在 `ppocr_rec/cfg/dicts/`。

```bash
python tools/convert_paddle_rec.py \
  --pdparams path/to/best_accuracy.pdparams \
  --yaml ppocrv5_mobile.yaml \
  --out pretrained/ppocrv5_mobile.pt \
  --dict ppocr_rec/cfg/dicts/ppocrv5_dict.txt
```

转换得到的 `.pt` 可作为 `pretrained=` 微调，或 `OCR("pretrained/ppocrv5_mobile.pt")` 直接预测。分类层宽度与新字典不一致时会自动跳过。加 `--strict-report` 可打印未匹配的参数名。详见 [docs/usage.md](docs/usage.md#9-从-paddleocr-转预训练)。

## 特别鸣谢

本仓库基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 的识别配方与数据处理，采用 PyTorch 实现，许可证为 [Apache 2.0](LICENSE)。接口组织参考了 [Ultralytics](https://github.com/ultralytics/ultralytics) 官方库。感谢两个项目的开源贡献。
