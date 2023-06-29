__constant sampler_t sampler = 
  CLK_NORMALIZED_COORDS_FALSE |
  CLK_FILTER_NEAREST          |
  CLK_ADDRESS_CLAMP_TO_EDGE;

typedef struct {int2 pos; float4 pix;} pipe_data_t;

__kernel void consumer(
  __read_only pipe pipe_data_t inputPipe,
  __write_only image2d_t outputImage,
  int totalPixels,
  float theta
) {
  pipe_data_t pipe_data;
  /* Loop to process all pixels from the producer kernel */
  for (int countPixels = 0; countPixels < totalPixels; countPixels++)
  {
    /* Keep trying to read a data from the pipe
     * until one becomes available */
    while(read_pipe(inputPipe, &pipe_data));
    //printf("pos:%d,%d, pix:%.2f,%.2f,%.2f", pipe_data.pos.x, pipe_data.pos.y, pipe_data.pix.x, pipe_data.pix.y, pipe_data.pix.z);
    /* Get coordinates */
    int posx = pipe_data.pos.x;
    int posy = pipe_data.pos.y;
    /* Get image Size */
    int imageWidth = get_image_width(outputImage);
    int imageHeight = get_image_height(outputImage);
    /* Compute image center */
    float x0 = imageWidth / 2.0f;
    float y0 = imageHeight / 2.0f;
    /* Compute the location relative to the image center */
    int xprime = posx - x0;
    int yprime = posy - y0;
    /* Compute sine and cosine */
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    /* Compute the output location */
    float2 writeCoord;
    writeCoord.x = xprime * cosTheta - yprime * sinTheta + x0;
    writeCoord.y = xprime * sinTheta + yprime * cosTheta + y0;
    /* Write the output image */
    write_imagef(outputImage, (int2)((int)writeCoord.x, (int)writeCoord.y), pipe_data.pix);
  }
}