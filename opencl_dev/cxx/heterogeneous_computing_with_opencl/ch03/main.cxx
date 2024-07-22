#include <CL/opencl.h>
#include <stdio.h>

int	main()
{
    const int elements = 2048;
    size_t datasize = sizeof(int) * elements;

    int *A = (int *)malloc(datasize);
    int *B = (int *)malloc(datasize);
    int *C = (int *)malloc(datasize);

    for (int i = 0; i < elements; ++i){
        A[i] = i;
        B[i] = i;
    }
    
    cl_int status;

    cl_uint numPlatforms = 0;
    cl_uint numDevices = 0;
    cl_platform_id *platforms = NULL;
    cl_device_id *devices = NULL;

    status = clGetPlatformIDs(0, NULL, &numPlatforms);

    platforms = (cl_platform_id *)malloc(numPlatforms * sizeof(cl_platform_id));
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    
    status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);

    devices = (cl_device_id *)malloc(numDevices * sizeof(cl_device_id));
    status = clGetDeviceIDs(platforms[1], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);

    cl_context context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);

    cl_command_queue cmdQueue = clCreateCommandQueueWithProperties(context, devices[0], 0, &status);

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, datasize, NULL, &status);
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, datasize, NULL, &status);

    status = clEnqueueWriteBuffer(cmdQueue, bufA, CL_TRUE, 0, datasize, A, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(cmdQueue, bufB, CL_TRUE, 0, datasize, B, 0, NULL, NULL);

    FILE* pFile = fopen("vecadd_kernel.cl", "r");
    if (pFile == NULL) {
        printf("open file error\n");
        return -1;
    }

    fseek(pFile, 0, SEEK_END);
    size_t sourceSize = ftell(pFile);
    char* programSource = (char *)malloc(sourceSize + 1);

    rewind(pFile);
    int n = fread(programSource, 1, sourceSize, pFile);
    fclose(pFile);

    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&programSource, &sourceSize, &status);
    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "vecadd", &status);
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufA);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufB);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufC);

    size_t indexSpaceSize[1], workGroupSize[1];
    indexSpaceSize[0] = datasize / sizeof(int);
    workGroupSize[0] = 256;

    status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, indexSpaceSize, workGroupSize, 0, NULL, NULL);
    status = clEnqueueReadBuffer(cmdQueue, bufC, CL_TRUE, 0, datasize, C, 0, NULL, NULL);

    printf("result:\n");
    for (int i = 0; i < elements; ++i){
        printf("%04d\t", C[i]);
    }
    printf("\n");

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseContext(context);

    free(devices);
    free(platforms);
    free(A);
    free(B);
    free(C);
    free(programSource);
    return 0;
}
