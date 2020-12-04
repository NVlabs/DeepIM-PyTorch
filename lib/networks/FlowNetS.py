import torch
import torch.nn as nn
from fcn.config import cfg
from torch.nn.init import kaiming_normal_
from point_matching_loss.PMLoss import PMLoss

__all__ = [
    'flownets', 'flownets_bn', 'flownets_rgbd'
]


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )


def fc(in_planes, out_planes, relu=True):
    if relu:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Linear(in_planes, out_planes)


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]


def t_transform_old(T_src, T_delta, zoom_factor, num_classes):
    '''
    :param T_src: (x1, y1, z1)
    :param T_delta: (dx, dy, dz)
    :return: T_tgt: (x2, y2, z2)
    '''

    T_src = T_src.repeat(1, num_classes)
    factor_x = torch.unsqueeze(zoom_factor[:, 0], 1)
    factor_y = torch.unsqueeze(zoom_factor[:, 1], 1)

    vx_0 = torch.mul(T_delta[:, 0::3], factor_x.repeat(1, num_classes))
    vy_0 = torch.mul(T_delta[:, 1::3], factor_y.repeat(1, num_classes))

    vz = torch.div(T_src[:, 2::3], torch.exp(T_delta[:, 2::3]))
    vx = torch.mul(vz, torch.addcdiv(vx_0, 1.0, T_src[:, 0::3], T_src[:, 2::3]))
    vy = torch.mul(vz, torch.addcdiv(vy_0, 1.0, T_src[:, 1::3], T_src[:, 2::3]))

    T_tgt = torch.zeros_like(T_src)
    T_tgt[:, 0::3] = vx
    T_tgt[:, 1::3] = vy
    T_tgt[:, 2::3] = vz

    return T_tgt


def t_transform(T_src, T_delta, zoom_factor, num_classes):
    '''
    :param T_src: (x1, y1, z1)
    :param T_delta: (dx, dy, dz)
    :return: T_tgt: (x2, y2, z2)
    '''

    weight = 10.0
    T_src = T_src.repeat(1, num_classes)
    vz = torch.div(T_src[:, 2::3], torch.exp(T_delta[:, 2::3] / weight))
    vx = torch.mul(vz, torch.addcdiv(T_delta[:, 0::3] / weight, 1.0, T_src[:, 0::3], T_src[:, 2::3]))
    vy = torch.mul(vz, torch.addcdiv(T_delta[:, 1::3] / weight, 1.0, T_src[:, 1::3], T_src[:, 2::3]))

    T_tgt = torch.zeros_like(T_src)
    T_tgt[:, 0::3] = vx
    T_tgt[:, 1::3] = vy
    T_tgt[:, 2::3] = vz

    return T_tgt


def t_transform_depth(T_src, T_delta, num_classes):
    '''
    :param T_src: (x1, y1, z1)
    :param T_delta: (dx, dy, dz)
    :return: T_tgt: (x2, y2, z2)
    '''

    T_src = T_src.repeat(1, num_classes)
    T_tgt = torch.zeros_like(T_src)
    T_tgt = T_src + T_delta

    return T_tgt



class FlowNetS(nn.Module):
    expansion = 1

    def __init__(self, num_classes, batchNorm=True):
        super(FlowNetS,self).__init__()

        self.num_classes = num_classes
        self.batchNorm = batchNorm
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1 = conv(self.batchNorm, 256,  256)
        self.conv4   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1 = conv(self.batchNorm, 512,  512)
        self.conv5   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1 = conv(self.batchNorm, 512,  512)
        self.conv6   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.batchNorm,1024, 1024)

        self.fc6 = fc(8 * 10 * 1024, 256, relu=True)
        self.fc7 = fc(256, 256, relu=True)
        self.fcr = fc(256, 4 * num_classes, relu=False)
        self.fct = fc(256, 3 * num_classes, relu=False)
        self.pml = PMLoss()

        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, weights_rot, poses_src, poses_tgt, extents, points, zoom_factor):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        # quaternion and translation
        out_conv6_flatten = out_conv6.view(out_conv6.size(0), -1)
        out_fc6 = self.fc6(out_conv6_flatten)
        out_fc7 = self.fc7(out_fc6)
        out_fcr = self.fcr(out_fc7)
        quaternion = nn.functional.normalize(torch.mul(out_fcr, weights_rot))
        translation_delta = self.fct(out_fc7)
        if cfg.INPUT == 'COLOR' and cfg.TRAIN.T_TRANSFORM_DEPTH == False:
            translation = t_transform(poses_src[:, 6:], translation_delta, zoom_factor, self.num_classes)
        else:
            translation = t_transform_depth(poses_src[:, 6:], translation_delta, self.num_classes)

        loss_pose = self.pml(quaternion, translation, poses_src, poses_tgt, extents, points)

        # optical flow
        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return [flow2,flow3,flow4,flow5,flow6], loss_pose, quaternion, translation
        else:
            return quaternion, translation

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class FlowNetS_RGBD(nn.Module):
    expansion = 1

    def __init__(self, num_classes, batchNorm=True):
        super(FlowNetS_RGBD,self).__init__()

        self.num_classes = num_classes
        self.batchNorm = batchNorm

        # RGB branch
        self.conv1_color   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2_color   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3_color   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1_color = conv(self.batchNorm, 256,  256)
        self.conv4_color   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1_color = conv(self.batchNorm, 512,  512)
        self.conv5_color   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1_color = conv(self.batchNorm, 512,  512)
        self.conv6_color   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1_color = conv(self.batchNorm,1024, 1024)

        # depth branch
        self.conv1_depth   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2)
        self.conv2_depth   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2)
        self.conv3_depth   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2)
        self.conv3_1_depth = conv(self.batchNorm, 256,  256)
        self.conv4_depth   = conv(self.batchNorm, 256,  512, stride=2)
        self.conv4_1_depth = conv(self.batchNorm, 512,  512)
        self.conv5_depth   = conv(self.batchNorm, 512,  512, stride=2)
        self.conv5_1_depth = conv(self.batchNorm, 512,  512)
        self.conv6_depth   = conv(self.batchNorm, 512, 1024, stride=2)
        self.conv6_1_depth = conv(self.batchNorm,1024, 1024)

        # fusion
        self.fc6 = fc(8 * 10 * 1024, 256, relu=True)
        self.fc7 = fc(256, 256, relu=True)
        self.fcr = fc(256, 4 * num_classes, relu=False)
        self.fct = fc(256, 3 * num_classes, relu=False)
        self.pml = PMLoss()

        # optical flow
        self.deconv5 = deconv(1024,512)
        self.deconv4 = deconv(1026,256)
        self.deconv3 = deconv(770,128)
        self.deconv2 = deconv(386,64)

        self.predict_flow6 = predict_flow(1024)
        self.predict_flow5 = predict_flow(1026)
        self.predict_flow4 = predict_flow(770)
        self.predict_flow3 = predict_flow(386)
        self.predict_flow2 = predict_flow(194)

        self.upsampled_flow6_to_5 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow5_to_4 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x_color, x_depth, weights_rot, poses_src, poses_tgt, extents, points, zoom_factor):

        # color forward
        out_conv2_color = self.conv2_color(self.conv1_color(x_color))
        out_conv3_color = self.conv3_1_color(self.conv3_color(out_conv2_color))
        out_conv4_color = self.conv4_1_color(self.conv4_color(out_conv3_color))
        out_conv5_color = self.conv5_1_color(self.conv5_color(out_conv4_color))
        out_conv6_color = self.conv6_1_color(self.conv6_color(out_conv5_color))

        # depth forward
        out_conv2_depth = self.conv2_depth(self.conv1_depth(x_depth))
        out_conv3_depth = self.conv3_1_depth(self.conv3_depth(out_conv2_depth))
        out_conv4_depth = self.conv4_1_depth(self.conv4_depth(out_conv3_depth))
        out_conv5_depth = self.conv5_1_depth(self.conv5_depth(out_conv4_depth))
        out_conv6_depth = self.conv6_1_depth(self.conv6_depth(out_conv5_depth))

        # add fusion
        out_conv2 = out_conv2_color + out_conv2_depth
        out_conv3 = out_conv3_color + out_conv3_depth
        out_conv4 = out_conv4_color + out_conv4_depth
        out_conv5 = out_conv5_color + out_conv5_depth
        out_conv6 = out_conv6_color + out_conv6_depth

        # quaternion and translation
        out_conv6_flatten = out_conv6.view(out_conv6.size(0), -1)
        out_fc6 = self.fc6(out_conv6_flatten)
        out_fc7 = self.fc7(out_fc6)
        out_fcr = self.fcr(out_fc7)
        quaternion = nn.functional.normalize(torch.mul(out_fcr, weights_rot))
        translation_delta = self.fct(out_fc7)
        translation = t_transform(poses_src[:, 6:], translation_delta, zoom_factor, self.num_classes)
        loss_pose = self.pml(quaternion, translation, poses_src, poses_tgt, extents, points)

        # optical flow
        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = crop_like(self.upsampled_flow6_to_5(flow6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        flow5       = self.predict_flow5(concat5)
        flow5_up    = crop_like(self.upsampled_flow5_to_4(flow5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        flow4       = self.predict_flow4(concat4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return [flow2,flow3,flow4,flow5,flow6], loss_pose, quaternion, translation
        else:
            return quaternion, translation

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def flownets(num_classes, data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(num_classes, batchNorm=False)
    if data is not None:
        model_dict = model.state_dict()
        print('model keys')
        print('=================================================')
        for k, v in model_dict.items():
            print(k)
        print('=================================================')

        print('data keys')
        print('=================================================')
        for k, v in data['state_dict'].items():
            print(k)
        print('=================================================')

        pretrained_dict = {k: v for k, v in data['state_dict'].items() if k in model_dict and v.size() == model_dict[k].size()}
        print('load the following keys from the pretrained model')
        print('=================================================')
        for k, v in pretrained_dict.items():
            print(k)
        print('=================================================')
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)
    return model


def flownets_rgbd(num_classes, data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS_RGBD(num_classes, batchNorm=False)
    if data is not None:
        pretrained_dict = data['state_dict']
        model_dict = model.state_dict()

        print('model keys')
        print('=================================================')
        for k, v in model_dict.items():
            print(k)
        print('=================================================')

        print('data keys')
        print('=================================================')
        for k, v in data['state_dict'].items():
            print(k)
        print('=================================================')

        # construct the dictionary for update
        update_dict = dict()
        for mk, mv in model_dict.items():
            key = mk

            if key in pretrained_dict:
                update_dict[mk] = pretrained_dict[key]
            else:
                # remove color or depth in the model key
                pos = mk.find('_color')
                if pos > 0:
                    key = mk[:pos] + mk[pos+6:]

                pos = mk.find('_depth')
                if pos > 0:
                    key = mk[:pos] + mk[pos+6:]

                if key in pretrained_dict:
                    update_dict[mk] = pretrained_dict[key]

        print('use pretrained weights for the following layers')
        print(update_dict.keys())
        model_dict.update(update_dict) 
        model.load_state_dict(model_dict)

    return model


def flownets_bn(data=None):
    """FlowNetS model architecture from the
    "Learning Optical Flow with Convolutional Networks" paper (https://arxiv.org/abs/1504.06852)

    Args:
        data : pretrained weights of the network. will create a new one if not set
    """
    model = FlowNetS(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
