typedef struct {int2 pos; float4 pix;} pipe_data_t;

__kernel void producer(
  __read_only image2d_t inputImage,
  __write_only pipe pipe_data_t outputPipe,
  __constant float *filter,
  int filterWidth,
  sampler_t sampler
) {
  /* Store each work-item's unique row and col */
  int col = get_global_id(0);
  int row = get_global_id(1);
  /* Half the width of the filter is needed for indexing
   * memory later*/
  int halfWidth = filterWidth / 2;
  /* Used to hold the value of the output pixel */
  float4 sum = {0.0f, 0.0f, 0.0f, 0.0f};
  sum.w = 255.0f;
  /* Iterator for the filter */
  int filterIdx = 0;
  /* Each work-item iterates around its local area on the basis of the
   * size of the filter */
  int2 coords; // Coordinates for accessing the image
  /* Iterate the filter rows */
  for (int i = -halfWidth; i <= halfWidth; i++)
  {
    coords.y = row + i;
    /* Iterate over the filter columns */
    for (int j = -halfWidth; j <= halfWidth; j++)
    {
      coords.x = col + j;
      /* Read a pixel from the image */
      float4 pixel;
      pixel = read_imagef(inputImage, sampler, coords);
      sum.x += pixel.x * filter[filterIdx];
      sum.y += pixel.y * filter[filterIdx];
      sum.z += pixel.z * filter[filterIdx];
      filterIdx++;
    }
  }
  pipe_data_t pipe_data = {coords, sum};
  /* Write the output pixel to the pipe */
  write_pipe(outputPipe, &pipe_data);
}