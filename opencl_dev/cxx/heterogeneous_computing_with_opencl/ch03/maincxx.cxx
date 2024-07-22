#define __CL_ENABLE_EXCEPTIONS
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

int main()
{
  const int elements = 2048;
  size_t datasize = sizeof(int) * elements;
  int *A = new int[elements];
  int *B = new int[elements];
  int *C = new int[elements];

  for (int i = 0; i < elements; i++) {
    A[i] = i;
    B[i] = i;
  }

  try {
    // Query for platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // Get a list of devices on this platform
    std::vector<cl::Device> devices;
    platforms[1].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    
    // Create a context for the devices
    cl::Context context(devices);
    // Create a command-queue for the first device
    cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);
    
    // Create the memory buffers
    cl::Buffer bufferA = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
    cl::Buffer bufferB = cl::Buffer(context, CL_MEM_READ_ONLY, datasize);
    cl::Buffer bufferC = cl::Buffer(context, CL_MEM_WRITE_ONLY, datasize);
    
    // Copy the input data to the input buffers using the
    // command-queue for the first device
    queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, datasize, A);
    queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, datasize, B);
   
    // Read the program source
    std::ifstream sourceFile("vecadd_kernel.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
   
    // Create the program from the source code
    cl::Program program = cl::Program(context, source);
    // Build the program for the devices
    program.build(devices);
   
    // Create the kernel
    cl::Kernel vecadd_kernel(program, "vecadd");

    // Set the kernel arguments
    vecadd_kernel.setArg(0, bufferA);
    vecadd_kernel.setArg(1, bufferB);
    vecadd_kernel.setArg(2, bufferC);
    
    // Execute the kernel
    cl::NDRange gloabl(elements);
    cl::NDRange local(256);
    queue.enqueueNDRangeKernel(vecadd_kernel, cl::NullRange, gloabl, local);
    
    // Copy the output data back to the host
    queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, datasize, C);
    for (int i = 0; i < elements; ++i) {
        std::cout << C[i] << "\t";
    }
    std::cout << std::endl;
  } 
  catch(cl::Error error) {
    std::cout << error.what() << "(" << error.err() << ")" << std::endl;
  }

  delete [] A;
  delete [] B;
  delete [] C;

  return 0;
}