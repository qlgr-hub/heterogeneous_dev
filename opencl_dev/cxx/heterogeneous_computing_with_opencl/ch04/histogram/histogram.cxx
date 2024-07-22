#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>

#include <CL/opencl.hpp>

#include "../../common/utils/bmp-utils.hpp"

static const int HIST_BINS = 256;

int main()
{
    // load input image to host memory
    ImageBmp<int> image{};
    loadFromFile("../../common/images/cat.bmp", image);

    // Query for platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // Get a list of devices on this platform
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    // Create a context for the devices
    cl::Context context{ devices };

    // Create a command-queue for the first device
    cl::CommandQueue queue = cl::CommandQueue{ context, devices[0] };
    
    // Create input buffer and map image data
    const int imageElements{ int(image.m_width * image.m_height) };
    const size_t inputDataSize{ imageElements * sizeof(int) };
    cl::Buffer bufInputImage = cl::Buffer{ context, CL_MEM_READ_ONLY, inputDataSize };
    queue.enqueueWriteBuffer(bufInputImage, CL_TRUE, 0, inputDataSize, (const void*)(image.m_imageData.data()));

    // Create output buffer and initialize
    const size_t outputHistogramSize{ HIST_BINS * sizeof(int) };
    cl::Buffer bufOutputHistogram = cl::Buffer{ context, CL_MEM_WRITE_ONLY, outputHistogramSize };
    queue.enqueueFillBuffer(bufOutputHistogram, 0, 0, outputHistogramSize);

    // Read the program source
    std::ifstream sourceFile{ "histogram.cl" };
    std::string sourceCode{ std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()) };
    cl::Program::Sources source{ 1, sourceCode };

    // Create the program from the source code
    cl::Program program = cl::Program{ context, source };
    // Build the program for the devices
    program.build(devices);

    // Create the kernel
    cl::Kernel histogramKernel{ program, "histogram" };

    // Set the kernel arguments
    histogramKernel.setArg(0, bufInputImage);
    histogramKernel.setArg(1, sizeof(int), &imageElements);
    histogramKernel.setArg(2, bufOutputHistogram);

    // Execute the kernel
    cl::NDRange gloabl(1024);
    cl::NDRange local(64);
    queue.enqueueNDRangeKernel(histogramKernel, cl::NullRange, gloabl, local);

    // Copy the output data back to the host
    std::array<int, HIST_BINS> hOutputHistogram{};
    queue.enqueueReadBuffer(bufOutputHistogram, CL_TRUE, 0, outputHistogramSize, (void*)(hOutputHistogram.data()));

    // print output result
    for (const auto& out : hOutputHistogram) {
        std::cout << std::setw(5) << out << "\t";
    }
    std::cout << std::endl;

    return 0;
}