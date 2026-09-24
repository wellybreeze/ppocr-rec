from ppocr_rec.nn.modules.backbones.hgnet import PPHGNet_small
from ppocr_rec.nn.modules.backbones.pphgnetv2 import PPHGNetV2_B4
from ppocr_rec.nn.modules.backbones.lcnetv3 import PPLCNetV3
from ppocr_rec.nn.modules.backbones.lcnetv4 import PPLCNetV4
from ppocr_rec.nn.modules.backbones.mobilenet_v3 import MobileNetV3
from ppocr_rec.nn.modules.backbones.mv1_enhance import MobileNetV1Enhance
from ppocr_rec.nn.modules.backbones.svtrnet import SVTRNet

BACKBONES = {
    "MobileNetV3": MobileNetV3,
    "SVTRNet": SVTRNet,
    "PPLCNetV3": PPLCNetV3,
    "PPLCNetV4": PPLCNetV4,
    "MobileNetV1Enhance": MobileNetV1Enhance,
    "PPHGNet_small": PPHGNet_small,
    "PPHGNetV2_B4": PPHGNetV2_B4,
}


def register_backbone(name, cls):
    BACKBONES[name] = cls


def build_backbone(cfg: dict, in_channels: int = 3):
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name not in BACKBONES:
        raise KeyError(f"Unknown backbone {name}. Registered: {list(BACKBONES)}")
    cfg.setdefault("in_channels", in_channels)
    return BACKBONES[name](**cfg)
