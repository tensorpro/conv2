## Imports
import torch
from torch.nn import functional as F
from torch import nn
from torch.autograd import Variable
import tensorflow as tf
import numpy as np

sess = tf.Session()

## Thing
# from convert import torch2tf_conv

def tuplify(x):
    if isinstance(x, tuple):
        return x
    return (x,x)

def assign(module, param, val):
    modules.sess.run(param.assign(val))

def convert_conv_torch2tf(kernel):
    return np.transpose(kernel.data.cpu().numpy(), [2,3,1,0])

def convert_torchout(torchout):
    return torchout.data.numpy().transpose([0,2,3,1])

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class Module(object):
    def __call__(self,x):
        return self.forward(x).eval(session=self.sess)

    def assign(self, param, val):
        self.sess.run([param.assign(val)])

class Conv2d(Module):

    def __init__(self,inc, outc, kernel_size,
                 stride=1, padding='VALID', bias=False, sess=sess):
        sh,sw = tuplify(stride)
        kh,kw = tuplify(kernel_size)
        w = np.random.random([kh,kw,inc,outc]).astype(np.float32)
        self.sess=sess
        self.weight = Param(w,sess)
        self.stride=[1,sh,sw,1]
        self.padding=padding
        if bias:
            b = np.random.random(outc)
            self.bias = Param(b,sess)
        else:
            self.bias=None
        sess.run(tf.global_variables_initializer())

    def forward(self, x):
        x= tf.nn.conv2d(x, self.weight, self.stride, self.padding)
        if self.bias is not None:
            return x+self.bias
        return x

class Param(tf.Variable):

    def __init__(self, w,  sess, dtype=tf.float32):
        super(Param, self).__init__(w, dtype=dtype)
        self.sess = sess
    def set_value(self, val):
        return sess.run(self.assign(val))

    def get_value(self):
        return self.eval(self.sess)
class BatchNorm2d(Module):

    def __init__(self, num_features, eps=1e-5):#, w, b, ma, mv, eps=1e-5, sess=sess):
        w = b = mv = ma = np.zeros(num_features)
        self.sess=sess
        self.eps=eps
        self.num_features = num_features
        self.weight = Param(w,sess)
        self.bias = Param(b, sess)
        self.running_var = Param(mv, sess)
        self.running_mean = Param(ma, sess)
        self.sess.run(tf.global_variables_initializer())
    def forward(self, x):
        return tf.nn.batch_normalization(x, self.running_mean, self.running_var,
                                         self.bias, self.weight, self.eps)
        
    
def sync_conv(tf_conv, torch_conv, sess=sess):
    sess.run(tf_conv.weight.assign(convert_conv_torch2tf(torch_conv.weight)))
    sess.run(tf_conv.bias.assign((torch_conv.bias).cpu().data.numpy()))
    
def bn(torch_bn):
    tfbn = BatchNorm2d(torch_bn.num_features)
    tfbn.weight.set_value(torch_bn.weight.data.numpy())
    tfbn.bias.set_value(torch_bn.bias.data.numpy())
    tfbn.running_mean.set_value(torch_bn.running_mean.numpy())
    tfbn.running_var.set_value(torch_bn.running_var.numpy())
    return tfbn

def conv(tconv):
    padding = 'VALID' if tconv.padding == (0,0) else "SAME"
    has_bias = tconv.bias is not None
    cnv = Conv2d(tconv.in_channels, tconv.out_channels, tconv.kernel_size,
                 stride=tconv.stride,padding=padding, bias=has_bias)
    cnv.weight.set_value(convert_conv_torch2tf(tconv.weight))
    if has_bias:
        cnv.bias.set_value(tconv.bias.data.numpy())
    return cnv
    
    



def pt_out_conversion(pt_out):
    return np.transpose(pt_out.data.numpy(), [0,2,3,1])

def test_conv(N=5, stride=1, padding=0, k=3, in_channels=2, out_channels=3):
    print("N={0} Stride={1}, padding={2}, k={3}".format(N,stride,padding,k))
    inp = np.random.random((1,N,N,in_channels)).astype(np.float32)
    inp_tf = tf.convert_to_tensor(inp)
    inp_pt = Variable(torch.Tensor(inp.transpose([0,3,1,2])))
    c_pt = nn.Conv2d(in_channels,out_channels,
                     k, padding=padding,stride=stride, bias=False)
    # c_pt = conv3x3(2,3,1)
    c_tf = conv(c_pt)
    tf_out = c_tf(inp_tf)
    pt_out = convert_torchout(c_pt(inp_pt))
    # print(tf_out)
    # print(pt_out)
    print(np.all(np.isclose(pt_out, tf_out)))
    # print(tf_out).shape
    # print(pt_out).sphape
    # print(np.isclose(pt_out, tf_out))
    return c_tf, c_pt
    
# ctf, cpt = test_conv()
def test_bn():
    tbn = nn.BatchNorm2d(2).eval()
    tfbn = bn(tbn)
    print(pt_out_conversion(tbn(inp_pt)))
    print(tfbn(inp_tf))
    return tbn, tfbn

# tbn,tfbn = test_bn()

print('conversion works on odd input size')
test_conv(5)
print('Conversion works for basic even input size')
test_conv(6)
print('Conversion works for even input size with various stride')
test_conv(6, stride=2)
test_conv(6, stride=3)

print('Conversion works for odd input same padding')
test_conv(5, padding=1)
test_conv(5, padding=1)

print('Conversion works for even input same padding')
test_conv(6, padding=1)
test_conv(6, padding=1)


print('Conversion works for odd input same padding, strided even but NOT odd')
test_conv(5, padding=1, stride=2)
test_conv(5, padding=1, stride=3)
test_conv(9, padding=1, stride=4)
test_conv(9, padding=1, stride=5)

print('Conversion fails for even input same padding, strided')
test_conv(6, padding=1, stride=2)
test_conv(6, padding=1, stride=3)
