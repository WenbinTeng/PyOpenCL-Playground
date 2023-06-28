__constant sampler_t sampler = 
  CLK_NORMALIZED_COORDS_FALSE |
  CLK_FILTER_NEAREST          |
  CLK_ADDRESS_CLAMP;

__kernel void rotate (
  __read_only image2d_t inputImage,
  __write_only image2d_t outputImage,
  float theta
) {
  /* Get global ID for ouput coordinates */
  int posx = get_global_id(0);
  int posy = get_global_id(1);
  int imageWidth = get_image_width(inputImage);
  int imageHeight = get_image_height(inputImage);
  /* Compute image center */
  float x0 = imageWidth / 2.0f;
  float y0 = imageHeight / 2.0f;
  /* Compute the work-item's location relative
   * to the image center */
  int xprime = posx - x0;
  int yprime = posy - y0;
  /* Compute sine and cosine */
  float sinTheta = sin(theta);
  float cosTheta = cos(theta);
  /* Compute the input location */
  float2 readCoord;
  readCoord.x = xprime * cosTheta - yprime * sinTheta + x0;
  readCoord.y = xprime * sinTheta + yprime * cosTheta + y0;
  /* Read the input image */
  float4 readPixel;
  readPixel = read_imagef(inputImage, sampler, readCoord);
  /* Write the output image */
  write_imagef(outputImage, (int2)(posx, posy), readPixel);
}