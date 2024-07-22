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
    ImageBmp<float> image{};
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

    // Copy the host image data to the device
    std::array<size_t, 3> origin{ 0, 0, 0 }; // Offset within the image to copy from
    std::array<size_t, 3> region{ image.m_width, image.m_height, 1 }; // Elements to per dimension
    queue.enqueueWriteImage(
        inputImage,
        CL_TRUE,
        origin,
        region,
        0,
        0,
        (const void*)(image.m_imageData.data())
    );

    // Read the program source
    std::ifstream sourceFile{ "image-rotation.cl" };
    std::string sourceCode{ std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()) };
    cl::Program::Sources source{ 1, sourceCode };

    // Create the program from the source code
    cl::Program program = cl::Program{ context, source };
    // Build the program for the devices
    program.build(devices);

    // Create the kernel
    cl::Kernel rotationKernel{ program, "rotation" };

    /* Angle for rotation (degrees) */
    const float theta = 45.f;
    
    // Set the kernel arguments
    rotationKernel.setArg(0, inputImage);
    rotationKernel.setArg(1, outputImage);
    rotationKernel.setArg(2, sizeof(int), &(image.m_width));
    rotationKernel.setArg(3, sizeof(int), &(image.m_height));
    rotationKernel.setArg(4, sizeof(float), &theta);
    
    // Execute the kernel
    cl::NDRange gloabl{ image.m_width, image.m_height };
    queue.enqueueNDRangeKernel(rotationKernel, cl::NullRange, gloabl);

    // Copy the output data back to the host
    ImageBmp<float> rotatedImage;
    rotatedImage.m_width = image.m_width;
    rotatedImage.m_height = image.m_height;
    rotatedImage.m_imageData.resize(image.m_width * image.m_height);
    queue.enqueueReadImage(
        outputImage, 
        CL_TRUE, 
        origin,
        region,
        0,
        0,
        (void*)(rotatedImage.m_imageData.data())
    );

    // save rotated image to new file
    saveToFile("rotated-cat-cxx.bmp", rotatedImage, "../../common/images/cat.bmp");

    return 0;
}