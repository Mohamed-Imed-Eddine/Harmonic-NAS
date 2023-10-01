# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import json
import torch

from backbones.ofa.utils import download_url
from backbones.ofa.imagenet_classification.networks import get_net_by_name
from backbones.ofa.imagenet_classification.elastic_nn.networks import (
    OFAMobileNetV3,
)



__all__ = [
    "ofa_specialized",
    "ofa_net",
    "proxylessnas_net",
    "proxylessnas_mobile",
    "proxylessnas_cpu",
    "proxylessnas_gpu",
]


def ofa_specialized(net_id, pretrained=True):
    url_base = "https://hanlab.mit.edu/files/OnceForAll/ofa_specialized/"
    net_config = json.load(
        open(
            download_url(
                url_base + net_id + "/net.config",
                model_dir=".torch/ofa_specialized/%s/" % net_id,
            )
        )
    )
    net = get_net_by_name(net_config["name"]).build_from_config(net_config)

    image_size = json.load(
        open(
            download_url(
                url_base + net_id + "/run.config",
                model_dir=".torch/ofa_specialized/%s/" % net_id,
            )
        )
    )["image_size"]

    if pretrained:
        init = torch.load(
            download_url(
                url_base + net_id + "/init",
                model_dir=".torch/ofa_specialized/%s/" % net_id,
            ),
            map_location="cpu",
        )["state_dict"]
        net.load_state_dict(init)
    return net, image_size


def ofa_net(net_id, model_dir=".torch/ofa_nets", resolution=224, pretrained=True, in_ch=3, _type='orig'):
    if net_id == "ofa_mbv3_d234_e346_k357_w1.0":
        net = OFAMobileNetV3(
            _type= _type,
            in_ch = in_ch,
            resolution=resolution,
            dropout_rate=0,
            width_mult=1.0,
            ks_list=[3, 5, 7],
            expand_ratio_list=[3, 4, 6],
            depth_list=[2, 3, 4],
        )
    elif net_id == "ofa_mbv3_d234_e346_k357_w1.2":
        net = OFAMobileNetV3(
            _type= _type,
            in_ch = in_ch,
            resolution=resolution,
            dropout_rate=0,
            width_mult=1.2,
            ks_list=[3, 5, 7],
            expand_ratio_list=[3, 4, 6],
            depth_list=[2, 3, 4],
        )

    if pretrained:
        url_base = "https://hanlab.mit.edu/files/OnceForAll/ofa_nets/"
        init = torch.load(
            download_url(url_base + net_id, model_dir=model_dir),
            map_location="cpu",
        )["state_dict"]
        net.load_state_dict(init)
    return net
