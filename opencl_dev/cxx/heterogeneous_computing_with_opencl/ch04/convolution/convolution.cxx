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
        std::vector<cl::Device> devices;
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);

        // Create a context for the devices
        cl::Context context{ devices };

        // Create a command-queue for the first device
        cl::CommandQueue queue = cl::CommandQueue{ context, devices[0] };
        
        // Create the input image and initialize it using a
        // pointer to the image data on the host
        cl::ImageFormat imageFormat{ CL_R, CL_FLOAT };
        cl::Image2D inputImage = cl::Image2D { 
            context, 
            CL_MEM_READ_ONLY, 
            imageFormat, 
            image.m_width,
            image.m_height
        };

        // Create the ouput image
        cl::Image2D outputImage = cl::Image2D {
            context, 
            CL_MEM_WRITE_ONLY, 
            imageFormat, 
            image.m_width,
            image.m_height
        };

        // Create a buffer for the filter
        cl::Buffer bufFilter = cl::Buffer{ context, CL_MEM_READ_ONLY, filterSize };

        // Copy the host image data to the device
        std::array<size_t, 3> origin{ 0, 0, 0 };
        std::array<size_t, 3> region{ image.m_width, image.m_height, 1 };
        queue.enqueueWriteImage(
            inputImage,
            CL_TRUE,
            origin,
            region,
            0,
            0,
            (const void*)(image.m_imageData.data())
        );

        // Copy the filter to the buffer
        queue.enqueueWriteBuffer(bufFilter, CL_TRUE, 0, filterSize, (const void*)(gaussianBlurFilter.data()));

        // Create the sampler
        cl::Sampler sampler = cl::Sampler{ context, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST };

        // Read the program source
        std::ifstream sourceFile{ "convolution.cl" };
        std::string sourceCode{ std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()) };
        cl::Program::Sources source{ 1, sourceCode };

        // Create the program from the source code
        cl::Program program = cl::Program{ context, source };
        // Build the program for the devices
        program.build(devices);

        // Create the kernel
        cl::Kernel convolutionKernel{ program, "convolution" };
        
        // Set the kernel arguments
        convolutionKernel.setArg(0, inputImage);
        convolutionKernel.setArg(1, outputImage);
        convolutionKernel.setArg(2, bufFilter);
        convolutionKernel.setArg(3, gaussianBlurFilterWidth);
        convolutionKernel.setArg(4, sampler);
        
        // Execute the kernel
        cl::NDRange gloabl{ image.m_width, image.m_height };
        cl::NDRange local{ 8, 8 };
        queue.enqueueNDRangeKernel(convolutionKernel, cl::NullRange, gloabl , local);

        // Copy the output data back to the host
        ImageBmp<float> filteredImage;
        filteredImage.m_width = image.m_width;
        filteredImage.m_height = image.m_height;
        filteredImage.m_imageData.resize(image.m_width * image.m_height);
        queue.enqueueReadImage(
            outputImage, 
            CL_TRUE, 
            origin,
            region,
            0,
            0,
            (void*)(filteredImage.m_imageData.data())
        );

        // save rotated image to new file
        saveToFile("cat-filtered.bmp", filteredImage, inputImagePath);
    }
    catch(const cl::Error& err) {
        std::cout << err.what() << "(" << err.err() << ")" << std::endl;
    }

    return 0;
}