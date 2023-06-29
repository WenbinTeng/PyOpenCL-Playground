__kernel void ReLU(__global const float *ift, __global float *oft)
{
    int i = get_global_id(0);
    oft[i] = max((float)0, ift[i]);
}