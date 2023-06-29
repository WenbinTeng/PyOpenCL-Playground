__kernel void MaxPool2d(__global const float *ift, 
                        __global int *size, __global int *stride,
                        __global int *channel, 
                        __global int *input_height, __global int *input_width, 
                        __global int *output_height, __global int *output_width, 
                        __global float *oft)
{
    int sz = *size, sd = *stride;
    int C = *channel, Hi = *input_height, Wi = *input_width, Ho = *output_height, Wo = *output_width;
    int posc = get_global_id(0), posh = get_global_id(1), posw = get_global_id(2);
    
    int So = Ho*Wo, Si = Hi*Wi;
    int i = (posc*(So))+(posh*Wo)+(posw);
    int startX = posw*sd, startY = posh*sd;
    
    oft[i] = ift[(posc*(Si))+(startY*Wi)+startX];
    for(int y = 0; y < sz; y++) {
        for(int x = 0; x < sz; x++) {
            oft[i] = max(oft[i], ift[(posc*(Si))+((startY+y)*Wi)+(startX+x)]);
        }
    }
}