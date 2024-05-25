import torch

from brevitas.nn import QuantIdentity, QuantReLU, QuantLinear, QuantConv2d
#from brevitas.core.quant import QuantType

from torch import Tensor
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerChannelFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import TruncTo8bit
from brevitas.quant_tensor import QuantTensor

#from brevitas.quant.scaled_int import Int8ActPerTensorFloat

def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or issubclass(type(m), nn.Linear) or issubclass(type(m), nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

def conv3x3(in_planes, 
            out_planes, 
            kernel_size=3,
            weight_bit_width=8,
            weight_quant=None,
            stride=1,
            padding=0,
            bias=False,
            groups=1,
            dilation=1) -> QuantConv2d:
    return QuantConv2d( in_channels=in_planes,
                        out_channels=out_planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=dilation,
                        groups=groups,
                        bias=bias,
                        weight_quant=weight_quant,
                        weight_bit_width=weight_bit_width)

def conv1x1(in_planes, 
            out_planes, 
            kernel_size=1,
            weight_bit_width=8,
            weight_quant=Int8WeightPerChannelFloat, 
            stride=1,
            padding=0,
            bias=False) -> QuantConv2d:
    return QuantConv2d(in_channels=in_planes,
                        out_channels=out_planes,
                        kernel_size=kernel_size,
                        stride=stride,
                        #padding=padding,
                        bias=bias,
                        weight_quant=weight_quant,
                        weight_bit_width=weight_bit_width)

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, 
                 inplanes, 
                 planes, 
                 first_block,
                 stride=1, 
                 bias = False,
                 #downsample=None, 
                 act_bit_width=8,
                 weight_bit_width=8,
                 weight_quant=Int8WeightPerChannelFloat,
                 groups=1,
                 base_width=64, 
                 dilation=1, 
                 norm_layer=None,
                 shared_quant_act=None
                 ):
        
        super(Bottleneck, self).__init__()

        self.first_block = first_block
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        
        
        self.bn0 = nn.Sequential()
        if self.first_block == False:
            self.bn0 = nn.Sequential(
                norm_layer(self.expansion * width),
                # We add a ReLU activation here because FINN requires the same sign along residual adds
                qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True))

        self.conv1 = conv1x1(in_planes=inplanes, 
                             out_planes=width,
                             weight_bit_width=weight_bit_width,
                             weight_quant=weight_quant, 
                             #stride=stride,
                             #padding=0,
                             bias=bias)
        
        self.bn1 = norm_layer(width)
        
        self.relu1 = QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.conv2 = conv3x3(in_planes=width, 
                             out_planes=width,
                             weight_bit_width=weight_bit_width,
                             weight_quant=weight_quant, 
                             stride=stride,
                             groups=groups,
                             dilation=dilation)
        
        self.bn2 = norm_layer(width)
        
        self.relu2 = QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.conv3 = conv1x1(in_planes=width, 
                             out_planes=planes * self.expansion, 
                             weight_bit_width=weight_bit_width,
                             weight_quant=weight_quant)
        self.bn3 = norm_layer(planes * self.expansion)
        
        ##########################################################
        self.downsample = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes=inplanes, 
                        out_planes=self.expansion * planes,
                        weight_bit_width=weight_bit_width,
                        weight_quant=weight_quant, 
                        stride=stride,
                        padding=0,
                        bias=bias),
                norm_layer(self.expansion * planes),
                # We add a ReLU activation here because FINN requires the same sign along residual adds
                qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True))
            # Redefine shared_quant_act whenever shortcut is performing downsampling
            shared_quant_act = self.downsample[-1]
        else:
            self.downsample = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        
        
       
        
        if shared_quant_act is None:
            shared_quant_act = qnn.QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
            
        self.relu3 = QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)  # shared_quant_act

        self.relu_out = QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        #self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        #identity = x
        out = self.bn0(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)

        if self.downsample != None:
            x = self.downsample(x)
        
        assert isinstance(out, QuantTensor), "Perform add among QuantTensors"
        assert isinstance(x, QuantTensor), "Perform add among QuantTensors"

        out = out + x
        out = self.relu_out(out)

        return out

class ResNet(nn.Module):
    def __init__(self, 
                 block, 
                 layers, 
                 first_maxpool=False,
                 zero_init_residual=False,
                 num_classes=10, 
                 act_bit_width=8,
                 weight_bit_width=8,
                 round_average_pool=False,
                 last_layer_bias_quant=Int32Bias,
                 weight_quant=Int8WeightPerChannelFloat,
                 first_layer_weight_quant=Int8WeightPerChannelFloat,
                 last_layer_weight_quant=Int8WeightPerTensorFloat,
                 groups=1, 
                 width_per_group=64, 
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 bias=False):
        
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                            "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = QuantConv2d(in_channels=3,
                                 out_channels=self.inplanes,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3,
                                 bias=bias,
                                 weight_quant=weight_quant,
                                 weight_bit_width=weight_bit_width)
        self.maxpool = nn.MaxPool2d(kernel_size=3, 
                                    stride=2, 
                                    padding=1)
        self.bn1 = norm_layer(self.inplanes)
        
        shared_quant_act = QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        self.relu = shared_quant_act
        
        #self.relu1 = QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
        #self.relu2 = QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)

        self.layer1 = self._make_layer(block, 
                                       64, 
                                       layers[0],
                                       bias=bias,
                                       shared_quant_act=shared_quant_act)
        self.layer2 = self._make_layer(block, 
                                       128, 
                                       layers[1], 
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       bias=bias,
                                       shared_quant_act=shared_quant_act)
        self.layer3 = self._make_layer(block, 
                                       256, 
                                       layers[2], 
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       bias=bias,
                                       shared_quant_act=shared_quant_act)
        self.layer4 = self._make_layer(block, 
                                       512, 
                                       layers[3], 
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       bias=bias,
                                       shared_quant_act=shared_quant_act)
        
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = qnn.QuantAdaptiveAvgPool2d((1,1))
        #self.avgpool = qnn.QuantAvgPool2d((1,1))

        avgpool_float_to_int_impl_type = 'ROUND' if round_average_pool else 'FLOOR'
        self.final_pool = qnn.TruncAdaptiveAvgPool2d(#TruncAvgPool2d(
            kernel_size=4,
            trunc_quant=TruncTo8bit,
            float_to_int_impl_type=avgpool_float_to_int_impl_type,
            output_size=(1,1)
            )
        
        self.fc = QuantLinear(
                              in_features=512 * block.expansion,
                              out_features=num_classes,
                              weight_bit_width=weight_bit_width,
                              bias=True,
                              bias_quant=last_layer_bias_quant,
                              weight_quant=last_layer_weight_quant
                              )
        #self.quant_identity = QuantIdentity(return_quant_tensor=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
        

    def _make_layer(self, 
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilate=False,
                    bias=False,
                    act_bit_width=8,
                    shared_quant_act=None):
        
        norm_layer = self._norm_layer
        
        downsample = None
        
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
        """
        if stride != 1 or self.inplanes != (planes * block.expansion):
            downsample = nn.Sequential(
                conv1x1(in_planes=self.inplanes, 
                        out_planes=planes * block.expansion, 
                        weight_bit_width=8,
                        weight_quant=Int8WeightPerChannelFloat, 
                        stride=stride,
                        bias=bias
                        ),
                norm_layer(planes * block.expansion),
                QuantReLU(bit_width=act_bit_width, return_quant_tensor=True)
            )
         """

        layers = []
        
        layers.append(block(self.inplanes, 
                            planes=planes, 
                            stride=stride,    
                            bias = bias,
                            #downsample=downsample, 
                            act_bit_width=8,
                            weight_bit_width=8,
                            weight_quant=Int8WeightPerChannelFloat,
                            groups=self.groups,
                            base_width=self.base_width, 
                            dilation=previous_dilation, 
                            norm_layer=norm_layer,
                            shared_quant_act=shared_quant_act,
                            first_block = True))
        
        self.inplanes = planes * block.expansion
          
        for _ in range(1, blocks):
            shared_quant_act = layers[-1].relu_out
            layers.append(block(self.inplanes,
                                planes=planes,
                                bias = bias,
                                act_bit_width=8,
                                weight_bit_width=8,
                                weight_quant=Int8WeightPerChannelFloat,
                                groups=self.groups,
                                base_width=self.base_width, 
                                dilation=self.dilation, 
                                norm_layer=norm_layer,
                                shared_quant_act=shared_quant_act,
                                first_block = False))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):


        #print("self.conv1.quant_weight()", self.conv1.quant_weight().signed_t)
        #print("self.final_pool.quant_weight()", self.final_pool.quant_weight().signed_t)
        #print("self.bn1.quant_weight()", self.bn1.quant_weight().signed_t)
        #print("self.relu1.quant_weight()", self.relu1.quant_weight().signed_t)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.relu1(x)
        #x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.relu2(x)
        x = self.avgpool(x)
        # x = self.quant_identity(x)
        #x = self.final_pool(x)

        # x = torch.flatten(x, 1)
        # exit()

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet( block=block, 
                    layers=layers, 
                    first_maxpool=False,
                    zero_init_residual=False,
                    num_classes=1000, 
                    act_bit_width=8,
                    weight_bit_width=8,
                    round_average_pool=False,
                    last_layer_bias_quant=Int32Bias,
                    weight_quant=Int8WeightPerChannelFloat,
                    first_layer_weight_quant=Int8WeightPerChannelFloat,
                    last_layer_weight_quant=Int8WeightPerTensorFloat,
                    groups=1, 
                    width_per_group=64, 
                    replace_stride_with_dilation=None,
                    norm_layer=None,
                    bias=False)
    
    #(block, layers, **kwargs)
    if pretrained:
        model.apply(_weights_init)
    return model

def resnet50(pretrained=False, progress=True, **kwargs):
     return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)
