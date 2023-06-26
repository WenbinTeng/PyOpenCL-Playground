__kernel void Linear(__global const float *ift, 
                     __global float *weight, __global float *bias,
                     __global int *output_channel, __global int *input_channel,
                     __global float *oft)
{
    int Co = *output_channel, Ci = *input_channel;
    int posCo = get_global_id(0);
    
    oft[posCo] = bias[posCo];
    for(int k = 0; k < Ci; k++) {
        oft[posCo] += ift[k]*weight[posCo*Ci+k];
    }
}