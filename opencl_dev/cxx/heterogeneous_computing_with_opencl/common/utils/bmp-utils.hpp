#ifndef __BMPFUNCS_HPP__
#define __BMPFUNCS_HPP__

#include <vector>
#include <string_view>
#include <fstream>

template<typename PIXEL_T>
struct ImageBmp
{
    size_t m_width;
    size_t m_height;
    std::vector<PIXEL_T> m_imageData;

public:
    ImageBmp() : m_width{ 0 }, m_height{ 0 } 
    {
    }
};


template<typename PIXEL_T>
bool loadFromFile(std::string_view filePath, ImageBmp<PIXEL_T>& image)
{
    std::ifstream imageReader{ filePath.data(), std::ios::binary };
    if (!imageReader.is_open()) {
        return false;
    }

    int offset{ 0 };
    imageReader.seekg(10);
    imageReader.read((char*)&offset, sizeof(offset));
    if (imageReader.gcount() != sizeof(offset)) {
        return false;
    }

    int width{ 0 }, height{ 0 };
    imageReader.seekg(18);
    imageReader.read((char*)&width, sizeof(width));
    if (imageReader.gcount() != sizeof(width)) {
        return false;
    }

    imageReader.read((char*)&height, sizeof(height));
    if (imageReader.gcount() != sizeof(height)) {
        return false;
    }

    image.m_width = width;
    image.m_height = height;

    image.m_imageData.resize(width * height);
    imageReader.seekg(offset);

    int mod{ width % 4 };
    if (mod != 0) {
        mod = 4 - mod;
    }

    // NOTE bitmaps are stored in upside-down raster order.  So we begin
    // reading from the bottom left pixel, then going from left-to-right, 
    // read from the bottom to the top of the image.  For image analysis, 
    // we want the image to be right-side up, so we'll modify it here.
    //
    // First we read the image in upside-down
    //
    // Read in the actual image
    for(size_t i{ 0 }; i < height; ++i) {

        // add actual data to the image
        for(size_t j{ 0 }; j < width; ++j) {
            unsigned char pixel{ 0 };
            imageReader.read((char*)&pixel, sizeof(pixel));
            if (imageReader.gcount() != sizeof(pixel)) {
                return false;
            }

            image.m_imageData[i*width + j] = pixel;
        }

        // For the bmp format, each row has to be a multiple of 4, 
        // so I need to read in the junk data and throw it away
        for(size_t j{ 0 }; j < mod; ++j) {
            unsigned char pixel{ 0 };
            imageReader.read((char*)&pixel, sizeof(pixel));
            if (imageReader.gcount() != sizeof(pixel)) {
                return false;
            }
        }
    }

    // Then we flip it over
    for(size_t i{ 0 }; i < height/2; ++i) {
        int flipRow{ height - int(i+1) };
        for(size_t j{ 0 }; j < width; ++j) {
            PIXEL_T pixel = image.m_imageData[i*width + j];
            image.m_imageData[i*width + j] = image.m_imageData[flipRow*width + j];
            image.m_imageData[flipRow*width + j] = pixel;
        }
    }

    imageReader.close();
    return true;
}

template<typename PIXEL_T>
bool saveToFile(std::string_view filePath, ImageBmp<PIXEL_T>& image, std::string_view refFilePath)
{
    if (refFilePath.empty()) {
        return false;
    }

    std::ifstream imageReader{ refFilePath.data(), std::ios::binary };
    if (!imageReader.is_open()) {
        return false;
    }

    int offset{ 0 };
    imageReader.seekg(10);
    imageReader.read((char*)&offset, sizeof(offset));
    if (imageReader.gcount() != sizeof(offset)) {
        return false;
    }

    int width{ 0 }, height{ 0 };
    imageReader.seekg(18);
    imageReader.read((char*)&width, sizeof(width));
    if (imageReader.gcount() != sizeof(width)
        || width != image.m_width) {
        return false;
    }

    imageReader.read((char*)&height, sizeof(height));
    if (imageReader.gcount() != sizeof(height)
        || height != image.m_height) {
        return false;
    }

    unsigned char* header{ new unsigned char[offset] };
    if (header == NULL) {
        return false;
    }

    imageReader.seekg(0);
    imageReader.read((char*)header, offset);
    if (imageReader.gcount() != offset) {
        return false;
    }
    imageReader.close();

    // write to new file
    std::ofstream imageWriter{ filePath.data(), std::ios::binary };
    if (!imageWriter.is_open()) {
        return false;
    }

    imageWriter.write((const char*)header, offset);
    if (imageWriter.fail()) {
        return false;
    }

    // NOTE bmp formats store data in reverse raster order (see comment in
    // readImage function), so we need to flip it upside down here.  
    int mod{ width % 4 };
    if(mod != 0) {
        mod = 4 - mod;
    }

    for(int i{ height-1 }; i >= 0; --i) {
        for(int j{ 0 }; j < width; ++j) {
            unsigned char pixel = (unsigned char)image.m_imageData[i*width + j];
            imageWriter.write((const char*)&pixel, sizeof(pixel));
            if (imageWriter.fail()) {
                return false;
            }
        }

        // In bmp format, rows must be a multiple of 4-bytes.  
        // So if we're not at a multiple of 4, add junk padding.
        for(int j{ 0 }; j < mod; ++j) {
            unsigned char pixel = (unsigned char)(0);
            imageWriter.write((const char*)&pixel, sizeof(pixel));
            if (imageWriter.fail()) {
                return false;
            }
        }
    }

    imageWriter.close();
    return true;
}

#endif // !__BMPFUNCS_HPP__
