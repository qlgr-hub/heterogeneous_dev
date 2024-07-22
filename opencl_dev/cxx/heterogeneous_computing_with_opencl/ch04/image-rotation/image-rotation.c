/* System includes */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* OpenCL includes */
#include <CL/cl.h>

/* Utility functions */
#include "../../common/utils/utils.h"
#include "../../common/utils/bmp-utils.h"

int main(int argc, char *argv[])
{
    /* Host data */
  float *hInputImage = NULL;
  float *hOutputImage = NULL;

  /* Angle for rotation (degrees) */
  const float theta = 45.f;

  /* Allocate space for the input image and read the
   * data from disk */
  int imageRows;
  int imageCols;
  hInputImage = readBmpFloat("../../common/images/cat.bmp", &imageRows, &imageCols);
  const int imageElements = imageRows * imageCols;
  const size_t imageSize = imageElements * sizeof(float);

  /* Allocate space for the ouput image */
  hOutputImage = (float *)malloc(imageSize);
  if (!hOutputImage){ exit(-1); }
  
  /* Use this check the output of each API call */
  cl_int status;
  cl_uint numPlatforms = 0;
  cl_platform_id *platforms = NULL;

  status = clGetPlatformIDs(0, NULL, &numPlatforms);

  platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);
  check(status, __LINE__);
  
  /* Get the first device */
  cl_device_id device;
  status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 1, &device, NULL);
  check(status, __LINE__);
  
  /* Create a command-queue and associate it with the device */
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  check(status, __LINE__);

   /* Create a command-queue and associate it with the device */
  cl_command_queue cmdQueue;
  cmdQueue = clCreateCommandQueueWithProperties(context, device, 0, &status);
  check(status, __LINE__);

   /* The image descriptor describes how the data will be stored
   * in memory. This descriptor initializes a 2D image with no pitch */
  cl_image_desc desc;
  desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  desc.image_width = imageCols;
  desc.image_height = imageRows;
  desc.image_depth = 0;
  desc.image_array_size = 0;
  desc.image_row_pitch = 0;
  desc.image_slice_pitch = 0;
  desc.num_mip_levels = 0;
  desc.num_samples = 0;
  desc.buffer = NULL;

  /* The image format describes the properties of each pixel */
  cl_image_format format;
  format.image_channel_order = CL_R; // single channel
  format.image_channel_data_type = CL_FLOAT;

  /* Create the input image and initialize it using a
   * pointer to the image data on the host */
  cl_mem inputImage = clCreateImage(context, CL_MEM_READ_ONLY, &format, &desc, NULL, NULL);

  /* Create the ouput image */
  cl_mem outputImage = clCreateImage(context, CL_MEM_WRITE_ONLY, &format, &desc, NULL, NULL);

  /* Copy the host image data to the device */
  size_t origin[3] = {0,0,0}; // Offset within the image to copy from
  size_t region[3] = {imageCols, imageRows, 1}; // Elements to per dimension
  clEnqueueWriteImage(cmdQueue, inputImage, CL_TRUE,
    origin, region, 0 /* row-pitch */, 0 /* slice-pitch */, hInputImage, 0, NULL, NULL);
  
  /* Create a program with source code */
  char *programSource = readFile("image-rotation.cl");
  size_t programSourceLen = strlen(programSource);
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&programSource, &programSourceLen, &status);
  check(status, __LINE__);
  
  /* Build (compile) the program for the device */
  status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  if (status != CL_SUCCESS){
    printCompilerError(program, device);
    exit(-1);
  }

  /* Create the kernel */
  cl_kernel kernel;
  kernel = clCreateKernel(program, "rotation", &status);
  check(status, __LINE__);

  /* Set the kernel arguments */
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputImage);
  status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputImage);
  status |= clSetKernelArg(kernel, 2, sizeof(int), &imageCols);
  status |= clSetKernelArg(kernel, 3, sizeof(int), &imageRows);
  status |= clSetKernelArg(kernel, 4, sizeof(float), &theta);
  check(status, __LINE__);

  /* Define the index space and work-group size */
  size_t globalWorkSize[2];
  globalWorkSize[0] = imageCols;
  globalWorkSize[1] = imageRows;
  size_t localWorkSize[2];
  localWorkSize[0] = 8;
  localWorkSize[1] = 8;

  /* Enqueue the kernel for execution */
  status = clEnqueueNDRangeKernel(cmdQueue, kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
  check(status, __LINE__);

  /* Copy the output data back to the host */
  status = clEnqueueReadImage(cmdQueue, outputImage, CL_TRUE, origin, region, 0 /* row-pitch */, 0 /* slice-pitch */, hOutputImage, 0, NULL, NULL);
  check(status, __LINE__);

  /* Write the output image to file */
  writeBmpFloat(hOutputImage, "rotated-cat.bmp", imageRows, imageCols, "../../common/images/cat.bmp");

  /* Free OpenCL resources */
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(cmdQueue);
  clReleaseMemObject(inputImage);
  clReleaseMemObject(outputImage);
  clReleaseContext(context);

  /* Free host resources */
  free(hInputImage);
  free(hOutputImage);
  free(programSource);

  return 0;
}