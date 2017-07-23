from __future__ import print_function, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class TConv2d(nn.Conv2d):
    """
    Tensorflow-like conv2d
    """

    def __init__(self, img_w, img_h, in_channels, out_channels,
                 kernel_size, stride=1, padding='VALID',
                 dilation=1, groups=1, bias=False):

        tf_padding = padding.lower()
        assert (tf_padding == 'valid' or tf_padding == 'same'), "Padding must be VALID or SAME"

        if tf_padding == "same":
            img_w, img_h, pad_t, pad_b, pad_l, pad_r = utils.calc_padding(
                img_w, img_h, kernel_size, kernel_size, stride, stride)
            print((img_w, img_h, pad_t, pad_b, pad_l, pad_r))
        else:
            pad_t = 0
            pad_l = 0

        super(TConv2d, self).__init__(in_channels, out_channels,
                                      kernel_size, stride=stride, padding=(int(pad_t), int(pad_l)),
                                      dilation=dilation, groups=groups, bias=bias)

        if tf_padding == 'same':
            if pad_t != pad_b:
                f_pad_t = 1
            else:
                f_pad_t = 0

            if pad_l != pad_r:
                f_pad_r = 1
            else:
                f_pad_r = 0

            self.f_padding = (0, f_pad_t, 0, f_pad_r)
        else:
            self.f_padding = (0, 0, 0, 0)

    def forward(self, x):
        x = F.pad(x, self.f_padding)
        print(self.f_padding)
        return super(TConv2d, self).forward(x)
