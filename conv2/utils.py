import math


def calc_padding(in_height, in_width,
                 filter_height, filter_width,
                 stride_h, stride_w):
    """
    Calculate padding when the 'SAME' padding is applied for Tensorflow
    """
    out_height = math.ceil(float(in_height) / float(stride_h))
    out_width = math.ceil(float(in_width) / float(stride_w))

    pad_along_height = max((out_height - 1) * stride_h +
                           filter_height - in_height, 0)
    pad_along_width = max((out_width - 1) * stride_w +
                          filter_width - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return (out_height, out_width, pad_top, pad_bottom, pad_left, pad_right)
