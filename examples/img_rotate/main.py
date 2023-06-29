#!/usr/bin/env python

import numpy as np
import pyopencl as cl
from PIL import Image

import os
os.chdir("./chapters/img_rotate/")

theta = 3.14159
img_src = Image.open("./lenna.png")
img_col, img_row = img_src.size
img_input_np = np.asarray(img_src).astype(np.uint8)
img_output_np = np.zeros_like(img_input_np).astype(np.uint8)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

img_fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
input_buf = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, format=img_fmt, shape=img_src.size, hostbuf=img_input_np)
output_buf = cl.Image(ctx, mf.WRITE_ONLY, format=img_fmt, shape=img_src.size)

with open("./knl_rotate.cl", "r") as f:
    prg_src = f.read()
    prg = cl.Program(ctx, prg_src).build()
    knl = prg.rotate
    knl(queue, img_src.size, (8, 8), input_buf, output_buf, np.float32(theta))

cl.enqueue_copy(queue, img_output_np, output_buf, origin=(0,0,0), region=(img_col, img_row, 1), row_pitch=0, slice_pitch=0)
img_dst = Image.frombytes("RGBA", img_src.size, img_output_np.tobytes())
img_dst.save("./lenna_rotate.png")
img_dst.show()