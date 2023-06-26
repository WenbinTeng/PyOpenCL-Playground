__kernel void Conv2d(__global const float *ift, 
                     __global float *weight, __global float *bias,
                     __global int *output_channel, __global int *output_height, __global int *output_width,
                     __global int *input_channel, __global int *input_height, __global int *input_width,
                     __global int *feature_height, __global int *feature_width,
                     __global float *oft)
{
    int Ci = *input_channel, Hi = *input_height, Wi = *input_width;
    int Co = *output_channel, Ho = *output_height, Wo = *output_width;
    int Hf = *feature_height, Wf = *feature_width;
    int posc = get_global_id(0), posh = get_global_id(1), posw = get_global_id(2);
    int Si = Wi*Hi, So = Wo*Ho, Sf = Wf*Hf;
    int Vf = Sf*Ci;
    int i = posc*(So) + (posh*Wo+posw);
    
    oft[i] = bias[posc];
    for(int l = 0; l < Hf; l++) {
        for(int m = 0; m < Wf; m++) {
            for(int n = 0; n < Ci; n++) {
                oft[i] += ift[(n*Si)+((posh+l)*Wi)+(posw+m)]*weight[(posc*Vf)+(n*Sf)+(l*Wf)+(m)];
            }
        }
    }
}