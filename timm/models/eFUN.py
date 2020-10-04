""" PyTorch eFUN family
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .efficientnet_blocks import round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from .efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights
from .helpers import load_pretrained, adapt_model_from_file
from .layers import SelectAdaptivePool2d, create_conv2d
from .registry import register_model

__all__ = ['EfficientFUN']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (192, 28, 28), 'pool_size': (1, 1),
        'crop_pct': 0.875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'eFUN': _cfg(),
    'eFUN-L': _cfg()
}

_DEBUG = False


class EfficientFUN(nn.Module):

    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3,
                 channel_multiplier=1.0, channel_divisor=8, channel_min=None,
                 output_stride=32, pad_type='', act_layer=nn.ReLU, drop_rate=0., drop_path_rate=0.,
                 se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None, global_pool='avg'):
        super(EfficientFUN, self).__init__()
        norm_kwargs = norm_kwargs or {}

        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate

        self.conv_stem = nn.Sequential()
        self.bn1 = nn.Sequential()
        self.act1 = nn.Sequential()
        self._in_chs = 192

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            channel_multiplier, channel_divisor, channel_min, output_stride, pad_type, act_layer, se_kwargs,
            norm_layer, norm_kwargs, drop_path_rate, verbose=_DEBUG)
        self.blocks = nn.Sequential(*builder(self._in_chs, block_args))
        self.feature_info = builder.features
        self._in_chs = builder.in_chs

        # Head + Pooling
        self.conv_head = create_conv2d(self._in_chs, self.num_features, 1, padding=pad_type)
        self.bn2 = norm_layer(self.num_features, **norm_kwargs)
        self.act2 = act_layer(inplace=True)
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)

        # Classifier
        self.classifier = nn.Linear(self.num_features * self.global_pool.feat_mult(), self.num_classes)

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1, self.act1]
        layers.extend(self.blocks)
        layers.extend([self.conv_head, self.bn2, self.act2, self.global_pool])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        if num_classes:
            num_features = self.num_features * self.global_pool.feat_mult()
            self.classifier = nn.Linear(num_features, num_classes)
        else:
            self.classifier = nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return self.classifier(x)


def _create_efun_model(model_kwargs, default_cfg, pretrained=False):
    load_strict = True
    model = EfficientFUN(**model_kwargs)
    model.default_cfg = default_cfg
    if pretrained:
        load_pretrained(
            model,
            default_cfg,
            num_classes=model_kwargs.get('num_classes', 0),
            in_chans=model_kwargs.get('in_chans', 3),
            strict=load_strict)
    return model


def _gen_efun(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    # arch_def = [
    #     ['ir_r3_k5_s1_e4_c128_se0.25'],
    #     ['ir_r6_k5_s2_e4_c160_se0.25'],
    #     ['ir_r1_k3_s1_e6_c192_se0.25'],
    # ]
    variant_to_arch = {
        "eFUN":
            [
                ['ir_r3_k5_s1_e4_c128_se0.25'],
                ['ir_r6_k5_s2_e4_c160_se0.25'],
                ['ir_r1_k3_s1_e6_c192_se0.25'],
            ],
        "eFUN-L":
            [
                ['ir_r3_k5_s1_e4_c144_se0.25'],
                ['ir_r2_k5_s1_e5_c180_se0.25'],
                ['ir_r5_k5_s2_e4_c180_se0.25'],
                ['ir_r2_k3_s1_e6_c216_se0.25'],
            ]

    }
    arch_def = variant_to_arch[variant]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_channels(1280, channel_multiplier, 8, None),
        channel_multiplier=channel_multiplier,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_kwargs=resolve_bn_args(kwargs),
        **kwargs,
    )
    model = _create_efun_model(model_kwargs, default_cfgs[variant], pretrained)
    return model


@register_model
def efun(pretrained=False, **kwargs):
    """ DCTEfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efun(
        'eFUN', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model



@register_model
def efun_l(pretrained=False, **kwargs):
    """ DCTEfficientNet-B0 """
    # NOTE for train, drop_rate should be 0.2, drop_path_rate should be 0.2
    model = _gen_efun(
        'eFUN-L', channel_multiplier=1.0, depth_multiplier=1.0, pretrained=pretrained, **kwargs)
    return model
