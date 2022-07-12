from collections import OrderedDict

from torch import nn

from smoke.modeling import registry
from . import dla
from . import resnet

@registry.BACKBONES.register("DLA-34-DCN")
def build_dla_backbone(cfg):
    body = dla.DLA(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
    return model

@registry.BACKBONES.register("RESNET")
def build_resnet_backbone(cfg):
    body = resnet.get_resnet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
    return model


def build_backbone(cfg):
    if cfg.MODEL.BACKBONE.CONV_BODY.split('-')[0] == "RESNET":
        CONV_BODY = cfg.MODEL.BACKBONE.CONV_BODY.split('-')[0]
    else:
        CONV_BODY = cfg.MODEL.BACKBONE.CONV_BODY
    assert CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[CONV_BODY](cfg)
