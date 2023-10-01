
import yaml
from typing import List
import numpy as np

class EfficiencyEstimator(object):
    def __init__(self, fname, supernet):

        with open(fname, 'r') as fp:
            self.lut = yaml.safe_load(fp)

        self.supernet = supernet

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return '_'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        else:
            return TypeError

    def predict(self, ltype: str, _input, in_ch=None, out_ch=None, mid_ch=None, 
                expand=None, kernel=None, stride=None, use_se=None, use_bn=None,
                dim=None, heads=None):
        if(self.supernet in ['alphanet', 'attentivenet', 'nasvit']):
            infos = [ltype, 'input:%s' % self.repr_shape(_input)]
            if ltype in ('MBInvertedConvLayer',):
                assert None not in (expand, kernel, stride, in_ch, mid_ch, out_ch, use_se)
                infos += ['in_ch:%d' % in_ch, 'out_ch:%d' % out_ch, 'mid_ch:%d' % mid_ch, 
                          'expand:%d' % expand, 'kernel:%d' % kernel, 'stride:%d' % stride, 
                          'use_se:%d' % use_se]
            elif ltype in ('DynamicSwinTransformerBlock',):
                assert None not in (dim, heads)
                infos += ['dim:%d' % dim, 'heads:%d' % heads]
            elif ltype in ('ConvBnActLayer',):
                assert None not in (kernel, stride, in_ch, out_ch, use_bn)
                infos += ['in_ch:%d' % in_ch, 'out_ch:%d' % out_ch, 
                          'kernel:%d' % kernel, 'stride:%d' % stride, 
                          'use_bn:%d' % use_bn]
            elif ltype in ('LinearLayer',):
                assert None not in (in_ch, out_ch)
                infos += ['in_ch:%d' % in_ch, 'out_ch:%d' % out_ch]
            else: 
                raise NotImplementedError
            key = '-'.join(infos)

        elif(self.supernet in ['ofa', 'proxyless']):
            infos = [ltype, 'input:%s' % self.repr_shape(_input)]
            if ltype in ('MBConvLayer',):
                mid_ch = 0 if mid_ch == None else mid_ch 
                assert None not in (expand, kernel, stride, in_ch, mid_ch, out_ch, use_se)
                infos += ['in_ch:%d' % in_ch, 'out_ch:%d' % out_ch, 'mid_ch:%d' % mid_ch, 
                          'expand:%d' % expand, 'kernel:%d' % kernel, 'stride:%d' % stride, 
                          'use_se:%d' % use_se]
            elif ltype in ('ConvLayer',):
                use_se = 0 if use_se == None else use_se 
                assert None not in (kernel, stride, in_ch, out_ch, use_se, use_bn)
                infos += ['in_ch:%d' % in_ch, 'out_ch:%d' % out_ch, 
                          'kernel:%d' % kernel, 'stride:%d' % stride, 
                          'use_se:%d' % use_se, 'use_bn:%d' % use_bn]
            elif ltype in ('LinearLayer',):
                assert None not in (in_ch, out_ch)
                infos += ['in_ch:%d' % in_ch, 'out_ch:%d' % out_ch]
            else: 
                raise NotImplementedError
            key = '-'.join(infos)
            # print("key : {} lat {} enrg {} ".format(key, self.lut[key]['lat'], self.lut[key]['enrg']))
        else: 
            raise NotImplementedError
    
        return self.lut[key]['lat'], self.lut[key]['enrg']
        # return "0.85", "8.5"
    
def look_up_ofa_proxy(net, lut, resolution=224, supernet='ofa',  num_channels=3):
    def _half(x, times=1):
        for _ in range(times):
            x = np.ceil(x / 2)
        return int(x)

    predicted_latency = 0
    predicted_enrg = 0

    # first_conv
    use_bn = 1 if net.first_conv.config['use_bn'] else 0
    se = 1 if net.first_conv.config['use_se'] else 0
    lat, enrg = lut.predict(
        'ConvLayer', 
        [1, num_channels, resolution, resolution], 
        in_ch=net.first_conv.config['in_channels'], 
        out_ch=net.first_conv.config['out_channels'], 
        kernel=net.first_conv.config['kernel_size'], 
        stride=net.first_conv.config['stride'],
        use_se=se, use_bn=use_bn)
    
    predicted_latency += float(lat)
    predicted_enrg += float(enrg)

    # blocks
    fsize = _half(resolution)
    for block in net.blocks:
        se = 1 if block.config['conv']['use_se'] else 0
        stride = block.config['conv']['stride']
        out_fz = _half(fsize) if stride > 1 else fsize
        lat, enrg = lut.predict(
            'MBConvLayer',
            [1, block.config['conv']['in_channels'],fsize, fsize],
            in_ch=block.config['conv']['in_channels'],
            out_ch=block.config['conv']['out_channels'],
            mid_ch=block.config['conv']['mid_channels'],
            expand=block.config['conv']['expand_ratio'],
            kernel=block.config['conv']['kernel_size'],
            stride=stride, 
            use_se=se
        )
        predicted_latency += float(lat)
        predicted_enrg += float(enrg)
        fsize = out_fz

    if(supernet == 'ofa'):
        use_bn = 1 if net.final_expand_layer.config['use_bn'] else 0
        se = 1 if net.final_expand_layer.config['use_se'] else 0
        lat, enrg = lut.predict(
            'ConvLayer', 
            [1, net.final_expand_layer.config['in_channels'], fsize, fsize], 
            in_ch=net.final_expand_layer.config['in_channels'], 
            out_ch=net.final_expand_layer.config['out_channels'], 
            kernel=net.final_expand_layer.config['kernel_size'],
            stride=net.final_expand_layer.config['stride'],
            use_bn=use_bn)
        
        predicted_latency += float(lat)
        predicted_enrg += float(enrg)
        
    fsize = 1
    use_bn = 1 if net.feature_mix_layer.config['use_bn'] else 0
    se = 1 if net.feature_mix_layer.config['use_se'] else 0
    lat, enrg = lut.predict(
        'ConvLayer', 
        [1, net.feature_mix_layer.config['in_channels'], fsize, fsize], 
        in_ch=net.feature_mix_layer.config['in_channels'], 
        out_ch=net.feature_mix_layer.config['out_channels'], 
        kernel=net.feature_mix_layer.config['kernel_size'],
        stride=net.feature_mix_layer.config['stride'],
        use_bn=use_bn)
    
    predicted_latency += float(lat)
    predicted_enrg += float(enrg)

    # classifier
    lat, enrg = lut.predict(
        'LinearLayer',
        str(net.classifier.config['in_features']),
        in_ch=net.classifier.config['in_features'],
        out_ch=net.classifier.config['out_features']
    )

    predicted_latency += float(lat)
    predicted_enrg += float(enrg)

    return predicted_latency, predicted_enrg
