__kernel void ReLU2d(__global const float *ift, __global float *oft,
                     __global int *channel, __global int *height, __global int *width)
{
    int c = *channel;
    int h = *height;
    int w = *width;
    int posc = get_global_id(0), posh = get_global_id(1), posw = get_global_id(2);
    int i = posc*(w*h) + (posh*w+posw);
    oft[i] = max((float)0, ift[i]);
}