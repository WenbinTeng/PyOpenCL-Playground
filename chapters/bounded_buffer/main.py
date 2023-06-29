import numpy as np
import pyopencl as cl
from PIL import Image

import os
os.chdir("./chapters/bounded_buffer/")

img_src = Image.open("./lenna.png")
img_col, img_row = img_src.size
img_input_np = np.asarray(img_src).astype(np.uint8)
img_output_np = np.zeros_like(img_input_np).astype(np.uint8)

theta = 3.14159
gaussian_blur_filter = np.array([
    1.0 / 273.0,  4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0, 1.0 / 273.0,
    4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
    7.0 / 273.0, 26.0 / 273.0, 41.0 / 273.0, 26.0 / 273.0, 7.0 / 273.0,
    4.0 / 273.0, 16.0 / 273.0, 26.0 / 273.0, 16.0 / 273.0, 4.0 / 273.0,
    1.0 / 273.0,  4.0 / 273.0,  7.0 / 273.0,  4.0 / 273.0, 1.0 / 273.0
]).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
mf = cl.mem_flags

img_fmt = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
img_spl = cl.Sampler(ctx, False, cl.addressing_mode.CLAMP_TO_EDGE, cl.filter_mode.LINEAR)
filter_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=gaussian_blur_filter)
pipe = cl.Pipe(ctx, flags=0, packet_size=6, max_packets=img_col*img_row)
input_buf = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, format=img_fmt, shape=img_src.size, hostbuf=img_input_np)
output_buf = cl.Image(ctx, mf.WRITE_ONLY, format=img_fmt, shape=img_src.size)

with open("./knl_producer.cl", "r") as f:
    prg_src = f.read()
    prg = cl.Program(ctx, prg_src).build("-cl-std=CL2.0")
    knl = prg.producer
    knl(queue, img_src.size, (8, 8), input_buf, pipe, filter_buf, np.int32(5), img_spl)

with open("./knl_consumer.cl", "r") as f:
    prg_src = f.read()
    prg = cl.Program(ctx, prg_src).build("-cl-std=CL2.0")
    knl = prg.consumer
    knl(queue, (1,1), (1,1), pipe, output_buf, np.int32(img_col*img_row), np.float32(theta))

cl.enqueue_copy(queue, img_output_np, output_buf, origin=(0,0,0), region=(img_col, img_row, 1), row_pitch=0, slice_pitch=0)
img_dst = Image.frombytes("RGBA", img_src.size, img_output_np.tobytes())
img_dst.save("./lenna_filter.png")
img_dst.show()