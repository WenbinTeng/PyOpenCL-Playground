__kernel void filter(
  __read_only image2d_t inputImage,
  __write_only image2d_t outputImage,
  __constant float *filter,
  int filterWidth,
  sampler_t sampler
) {
  /* Store each work-item's unique row and col */
  int col = get_global_id(0);
  int row = get_global_id(1);
  /* Half the width of the filter is needed for indexing
   * memory later */
  int halfWidth = filterWidth / 2;
  /* All accesses to images return data as four-element vectors
   * (i.e., float4) */
  float4 sum = {0.0f, 0.0f, 0.0f, 255.0f};
  /* Iterator for the filter */
  int filterIdx = 0;
  /* Each work-item iterates around its local area on the basis of the
   * size of the filter*/
  int2 coords; // Coordinates for accessing the image
  /* Iterate the filter rows*/
  for (int i = -halfWidth; i <= halfWidth; i++)
  {
    coords.y = row + i;
    /* Iterate over the filter columns */
    for (int j = -halfWidth; j <= halfWidth; j++)
    {
      coords.x = col + j;
      /*Read a pixel from the image */
      float4 pixel = read_imagef(inputImage, sampler, coords);
      sum.x += pixel.x * filter[filterIdx];
      sum.y += pixel.y * filter[filterIdx];
      sum.z += pixel.z * filter[filterIdx];
      filterIdx++;
    }
  }
  /* Copy the data to the output image */
  write_imagef(outputImage, (int2)(col, row), sum);
}