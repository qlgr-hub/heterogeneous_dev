#define CL_HPP_ENABLE_EXCEPTIONS

#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <string>
#include <iomanip>

#include <CL/opencl.hpp>

#include "../../common/utils/bmp-utils.hpp"

static const char* inputImagePath = "../../common/images/cat.bmp";

static const int gaussianBlurFilterWidth = 5;
static const std::array<float, gaussianBlurFilterWidth*gaussianBlurFilterWidth> gaussianBlurFilter{
    1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f, 1.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f, 1.0f / 273.0f
};

// static const std::array<float, gaussianBlurFilterWidth*gaussianBlurFilterWidth> gaussianBlurFilter{
//      3.0f / 273.0f, 12.0f / 273.0f,  21.0f / 273.0f, 12.0f / 273.0f,  3.0f / 273.0f,
//     12.0f / 273.0f, 48.0f / 273.0f,  78.0f / 273.0f, 48.0f / 273.0f, 12.0f / 273.0f,
//     21.0f / 273.0f, 78.0f / 273.0f, 123.0f / 273.0f, 78.0f / 273.0f, 21.0f / 273.0f,
//     12.0f / 273.0f, 48.0f / 273.0f,  78.0f / 273.0f, 48.0f / 273.0f, 12.0f / 273.0f,
//      3.0f / 273.0f, 12.0f / 273.0f,  21.0f / 273.0f, 12.0f / 273.0f,  3.0f / 273.0f
// };
//
// static const std::array<float, gaussianBlurFilterWidth*gaussianBlurFilterWidth> gaussianBlurFilter1{
//     2.0f / 159.0f,  4.0f / 159.0f,  5.0f / 159.0f,  4.0f / 159.0f, 2.0f / 159.0f,
//     4.0f / 159.0f,  9.0f / 159.0f, 12.0f / 159.0f,  9.0f / 159.0f, 4.0f / 159.0f,
//     5.0f / 159.0f, 12.0f / 159.0f, 15.0f / 159.0f, 12.0f / 159.0f, 5.0f / 159.0f,
//     4.0f / 159.0f,  9.0f / 159.0f, 12.0f / 159.0f,  9.0f / 159.0f, 4.0f / 159.0f,
//     2.0f / 159.0f,  4.0f / 159.0f,  5.0f / 159.0f,  4.0f / 159.0f, 2.0f / 159.0f
// };
//
// static const std::array<float, gaussianBlurFilterWidth*gaussianBlurFilterWidth> gaussianBlurFilter2{
//     20.0f / 159.0f,  40.0f / 159.0f,  50.0f / 159.0f,  40.0f / 159.0f, 20.0f / 159.0f,
//     40.0f / 159.0f,  90.0f / 159.0f, 120.0f / 159.0f,  90.0f / 159.0f, 40.0f / 159.0f,
//     50.0f / 159.0f, 120.0f / 159.0f, 150.0f / 159.0f, 120.0f / 159.0f, 50.0f / 159.0f,
//     40.0f / 159.0f,  90.0f / 159.0f, 120.0f / 159.0f,  90.0f / 159.0f, 40.0f / 159.0f,
//     20.0f / 159.0f,  40.0f / 159.0f,  50.0f / 159.0f,  40.0f / 159.0f, 20.0f / 159.0f
// };

static const int HIST_BINS = 256;

int main()
{
    size_t filterSize{ gaussianBlurFilterWidth * gaussianBlurFilterWidth * sizeof(float) };

    try {
        // load input image to host memory
        ImageBmp<float> image{};
        loadFromFile(inputImagePath, image);

        // Query for platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        
        // Get a list of devices on this platform
        std::vector<cl::Device> producerDevices;
        std::vector<cl::Device> consumerDevices;
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &consumerDevices);
        platforms[1].getDevices(CL_DEVICE_TYPE_ALL, &producerDevices);
        
        // Create a context for the devices
        std::vector<cl::Device> devices;
        devices.insert(devices.end(), producerDevices.begin(), producerDevices.end());
        devices.insert(devices.end(), consumerDevices.begin(), consumerDevices.end());
        cl::Context context{ devices };

        size_t contextDeviceCount = 0;
        context.getInfo(CL_CONTEXT_NUM_DEVICES, &contextDeviceCount);
        std::cout << contextDeviceCount << std::endl;

        // Create a command-queue for the first device
        cl::CommandQueue producerQueue{ context, devices[0] };
        cl::CommandQueue consumerQueue{ context, devices[1] };
        
        // Create the input image and initialize it using a
        // pointer to the image data on the host
        cl::ImageFormat imageFormat{ CL_R, CL_FLOAT };
        cl::Image2D inputImage{ 
            context, 
            CL_MEM_READ_ONLY, 
            imageFormat, 
            image.m_width,
            image.m_height
        };

        // Create a buffer for the filter
        cl::Buffer bufFilter{ context, CL_MEM_READ_ONLY, filterSize };

        // Create the pipe
        cl::Pipe pipe{ context, sizeof(float), static_cast<cl_uint>(image.m_width * image.m_height) };

        // Create output buffer and initialize
        const size_t outputHistogramSize{ HIST_BINS * sizeof(int) };
        cl::Buffer bufOutputHistogram{ context, CL_MEM_WRITE_ONLY, outputHistogramSize};
        
        // Copy the host image data to the device
        std::array<size_t, 3> origin{ 0, 0, 0 };
        std::array<size_t, 3> region{ image.m_width, image.m_height, 1 };
        producerQueue.enqueueWriteImage(
            inputImage,
            CL_TRUE,
            origin,
            region,
            0,
            0,
            (const void*)(image.m_imageData.data())
        );

        // Copy the filter to the buffer
        producerQueue.enqueueWriteBuffer(bufFilter, CL_TRUE, 0, filterSize, (const void*)(gaussianBlurFilter.data()));

        // Initialize the output istogram with zeros
        consumerQueue.enqueueFillBuffer(bufOutputHistogram, 0, 0, outputHistogramSize);

        // Read the program source
        std::ifstream sourceFile{ "producer-consumer.cl" };
        std::string sourceCode{ std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()) };
        cl::Program::Sources source{ 1, sourceCode };

        // Create the program from the source code
        cl::Program program = cl::Program{ context, source };

        // Build the program for the devices
        program.build(devices);

        // Create the kernel
        cl::Kernel producerKernel{ program, "producerKernel" };
        cl::Kernel consumerKernel{ program, "consumerKernel" };
        
        // Set the kernel arguments
        producerKernel.setArg(0, inputImage);
        producerKernel.setArg(1, pipe);
        producerKernel.setArg(2, bufFilter);
        producerKernel.setArg(3, gaussianBlurFilterWidth);

        consumerKernel.setArg(0, pipe);
        consumerKernel.setArg(1, static_cast<int>(image.m_imageData.size()));
        consumerKernel.setArg(2, bufOutputHistogram);
        
        // Execute the kernel
        cl::NDRange producerGloabl{ image.m_width, image.m_height };
        cl::NDRange producerLocal{ 8, 8 };
        producerQueue.enqueueNDRangeKernel(producerKernel, cl::NullRange, producerGloabl , producerLocal);

        cl::NDRange consumerGloabl{ 1 };
        cl::NDRange consumerLocal{ 1 };
        consumerQueue.enqueueNDRangeKernel(consumerKernel, cl::NullRange, consumerGloabl , consumerLocal);

        // Copy the output data back to the host
        std::array<int, HIST_BINS> hOutputHistogram{};
        consumerQueue.enqueueReadBuffer(bufOutputHistogram, CL_TRUE, 0, outputHistogramSize, (void*)(hOutputHistogram.data()));

        // print output result
        for (const auto& out : hOutputHistogram) {
            std::cout << std::setw(5) << out << "\t";
        }
        std::cout << std::endl;
    }
    catch(const cl::Error& err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}