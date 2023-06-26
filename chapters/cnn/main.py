#!/usr/bin/env python

import numpy as np
import pyopencl as cl
import torch
import torch.nn as nn
import torchvision.models as models
import time

import os
os.chdir("./chapters/cnn")

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

def Conv2dWrapper(input_numpy, kernel_weight_numpy, kernel_bias_numpy, padding = 0):
    weight_cpu = kernel_weight_numpy
    bias_cpu = kernel_bias_numpy

    Ci, Hi, Wi = input_numpy.shape
    input_cpu = np.zeros((Ci, Hi+2*padding, Wi+2*padding)).astype(np.float32)
    if padding > 0:
        input_cpu[:, padding:-padding, padding:-padding] = input_numpy
    else:
        input_cpu = input_numpy

    Ci, Hi, Wi     = input_cpu.shape
    Co, Ci, Hf, Wf = weight_cpu.shape
    Ho, Wo         = Hi - Hf + 1, Wi - Wf + 1

    output_cpu = np.zeros((Co,Ho,Wo)).astype(np.float32)
    input_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = input_cpu)

    kernel_weight_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = weight_cpu)
    kernel_bias_gpu   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = bias_cpu)

    output_channel_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Co))
    output_height_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Ho))
    output_width_gpu   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Wo))
    input_channel_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Ci))
    input_height_gpu   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Hi))
    input_width_gpu    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Wi))
    feature_height_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Hf))
    feature_width_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Wf))
    
    output_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, output_cpu.nbytes)

    prg_src = open("./knl_Conv2d.cl", "r").read()
    prg = cl.Program(ctx, prg_src).build()
    knl = prg.Conv2d
    knl(queue, output_cpu.shape, None, 
        input_gpu, 
        kernel_weight_gpu, kernel_bias_gpu,
        output_channel_gpu, output_height_gpu, output_width_gpu,
        input_channel_gpu, input_height_gpu, input_width_gpu,
        feature_height_gpu, feature_width_gpu,
        output_gpu)
    cl.enqueue_copy(queue, output_cpu, output_gpu)

    return output_cpu

def LinearWrapper(input_numpy, weight_numpy, bias_numpy):
    input_cpu = input_numpy
    weight_cpu = weight_numpy
    bias_cpu = bias_numpy

    Co, Ci = weight_cpu.shape

    output_cpu = np.zeros((Co,)).astype(np.float32)
    input_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = input_cpu)

    weight_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = weight_cpu)
    bias_gpu   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = bias_cpu)

    output_channel_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Co))
    input_channel_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Ci))

    output_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, output_cpu.nbytes)

    prg_src = open("./knl_Linear.cl", "r").read()
    prg = cl.Program(ctx, prg_src).build()
    knl = prg.Linear
    knl(queue, output_cpu.shape, None, 
        input_gpu, 
        weight_gpu, bias_gpu,
        output_channel_gpu, input_channel_gpu,
        output_gpu)
    cl.enqueue_copy(queue, output_cpu, output_gpu)

    return output_cpu

def MaxPool2dWrapper(input_numpy,size=2,stride=2):
    C, Hi, Wi = input_numpy.shape
    Ho, Wo = int(np.floor(Hi/stride)),int(np.floor(Wi/stride))
    row_remainder,col_remainder = Hi%stride, Wi%stride
    Ho += int(row_remainder!=0)
    Wo += int(col_remainder!=0)
    input_cpu = np.zeros((C, Hi+size-row_remainder, Wi+size-col_remainder)).astype(np.float32)
    input_cpu[:, :Hi, :Wi] = input_numpy

    C, Hi, Wi = input_cpu.shape

    output_cpu = np.zeros((C,Ho,Wo)).astype(np.float32)
    input_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = input_cpu)

    size_gpu   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(size))
    stride_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(stride))

    channel_gpu       = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(C))
    input_height_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Hi))
    input_width_gpu   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Wi))
    output_height_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Ho))
    output_width_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(Wo))

    output_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, output_cpu.nbytes)

    prg_src = open("./knl_MaxPool2d.cl", "r").read()
    prg = cl.Program(ctx, prg_src).build()
    knl = prg.MaxPool2d
    knl(queue, output_cpu.shape, None, 
        input_gpu, 
        size_gpu, stride_gpu,
        channel_gpu,
        input_height_gpu, input_width_gpu,
        output_height_gpu, output_width_gpu,
        output_gpu)
    cl.enqueue_copy(queue, output_cpu, output_gpu)

    return output_cpu

def ReLUWrapper(input_numpy):
    output_cpu = np.empty_like(input_numpy)

    input_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = input_numpy)
    output_gpu = cl.Buffer(ctx, mf.WRITE_ONLY, output_cpu.nbytes)
    
    if len(input_numpy.shape) == 3:
        channel_gpu = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(input_numpy.shape[0]))
        height_gpu  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(input_numpy.shape[1]))
        width_gpu   = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = np.int32(input_numpy.shape[2]))
        prg_src = open("./knl_ReLU2d.cl", "r").read()
        prg = cl.Program(ctx, prg_src).build()
        knl = prg.ReLU2d
        knl(queue, input_numpy.shape, None, input_gpu, output_gpu, channel_gpu, height_gpu, width_gpu)
    else:
        prg_src = open("./knl_ReLU.cl", "r").read()
        prg = cl.Program(ctx, prg_src).build()
        knl = prg.ReLU
        knl(queue, input_numpy.shape, None, input_gpu, output_gpu)
        
    cl.enqueue_copy(queue, output_cpu, output_gpu)

    return output_cpu

def eval_cpu():
    t1 = time.time()
    model = models.AlexNet()
    x = torch.from_numpy(np.random.randn(3, 224, 224).astype("float32"))
    x = model.features(x)
    x = model.avgpool(x)
    x = torch.flatten(x)
    x = model.classifier(x)
    t2 = time.time()
    print('CPU elapsed time: {}ms'.format(1000*(t2-t1)))
    
def eval_gpu():
    t1 = time.time()
    model = models.AlexNet()
    x = np.random.randn(3, 224, 224).astype("float32")
    
    conv2d_1_out    = Conv2dWrapper(x, model.features[0].weight.detach().numpy(), model.features[0].bias.detach().numpy(), padding=2)
    relu_1_out      = ReLUWrapper(conv2d_1_out)
    maxpool2d_1_out = MaxPool2dWrapper(relu_1_out, size=3, stride=2)
    
    conv2d_2_out    = Conv2dWrapper(maxpool2d_1_out, model.features[3].weight.detach().numpy(), model.features[3].bias.detach().numpy(), padding=2)
    relu_2_out      = ReLUWrapper(conv2d_2_out)
    maxpool2d_2_out = MaxPool2dWrapper(relu_2_out, size=3, stride=2)
    
    conv2d_3_out    = Conv2dWrapper(maxpool2d_2_out, model.features[6].weight.detach().numpy(), model.features[6].bias.detach().numpy(), padding=1)
    relu_3_out      = ReLUWrapper(conv2d_3_out)
    
    conv2d_4_out    = Conv2dWrapper(relu_3_out, model.features[8].weight.detach().numpy(), model.features[8].bias.detach().numpy(), padding=1)
    relu_4_out      = ReLUWrapper(conv2d_4_out)
    
    conv2d_5_out    = Conv2dWrapper(relu_4_out, model.features[10].weight.detach().numpy(), model.features[10].bias.detach().numpy(), padding=1)
    relu_5_out      = ReLUWrapper(conv2d_5_out)
    maxpool2d_5_out = MaxPool2dWrapper(relu_5_out, size=3, stride=2)
    
    x = torch.flatten(nn.AdaptiveAvgPool2d((6,6))(torch.from_numpy(maxpool2d_5_out))).detach().numpy()
    
    linear_6_out    = LinearWrapper(x, model.classifier[1].weight.detach().numpy(), model.classifier[1].bias.detach().numpy())
    relu_6_out      = ReLUWrapper(linear_6_out)
    
    linear_7_out    = LinearWrapper(relu_6_out, model.classifier[4].weight.detach().numpy(), model.classifier[4].bias.detach().numpy())
    relu_7_out      = ReLUWrapper(linear_7_out)
    
    linear_8_out    = LinearWrapper(relu_7_out, model.classifier[6].weight.detach().numpy(), model.classifier[6].bias.detach().numpy())
    
    x = linear_8_out
    t2 = time.time()
    print('GPU elapsed time: {}ms'.format(1000*(t2-t1)))
    
if __name__ == "__main__":
    eval_gpu()
    eval_cpu()
