/* Copyright 2026 Roman Kudinov. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#pragma once

#include <libocl/Definitions.hpp>
#include <libocl/Runtime.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <libocl/TextureBase.hpp>

namespace ocl
{
    /**
     * @class Texture2DArray
     * @brief Represents a 2D texture array in OpenCL. It is a more sophisticated memory type than RemoteBuffer, as it uses optimized texture memory.
     * When creating a Texture2DArray, specific image details such as width, height, depth, format, and access type are provided.
     * Within an OpenCL kernel, a Texture2DArray is not accessed as a global pointer to memory but rather as a sampler object,
     * and you must use image-specific functions to work with texture arrays.
     * Note that OpenCV support for Texture2DArray is minimal and straightforward, supporting only 1, 2, or 4 channels.
     */
    class Texture2DArray final : public ocl::TextureBase
    {
        friend class Texture2D;
        public:
            /**
             * @brief Constructor. Creates an uninitialized Texture2DArray with the specified width, height, depth, format, access type, and access mode.
             * @param runtime The Runtime context in which the texture array will be created.
             * @param width The width of each texture slice in pixels.
             * @param height The height of each texture slice in pixels.
             * @param depth The number of texture slices in the array (i.e., the depth of the array).
             * @param format The image format of the texture array (e.g., RGBA, etc.). See @ref ImageFormat for details.
             * @param type The access type for the texture array (e.g., ReadOnly, WriteOnly, ReadWrite).
             * @param access The access mode for the texture array (default is ReadWrite).
             */
			Texture2DArray(const Runtime& runtime, uint width, uint height, uint depth, ImageFormat format, AccessType type, AccessMode access = AccessMode::ReadWrite);

			/**
             * @brief Constructor. Creates a Texture2DArray from an array of OpenCV Mat objects. The data from the Mat objects will be uploaded to the texture array.
             * Each Mat in the array corresponds to a slice in the texture array, and all Mats must have the same width, height, and format.
             * @param runtime The Runtime context in which the texture array will be created.
             * @param mat_array A pointer to an array of OpenCV Mat objects from which to create the texture array. Each Mat corresponds to a slice in the texture array, and all Mats must have the same width, height, and format.
             * @param depth The number of texture slices in the array (i.e., the depth of the array). This should match the number of Mats in mat_array.
             * @param type The access type for the texture array (e.g., ReadOnly, WriteOnly, ReadWrite).
             * @param access The access mode for the texture array (default is ReadWrite).
             */
			Texture2DArray(const Runtime& runtime, const cv::Mat* mat_array, uint depth, AccessType type, AccessMode access = AccessMode::ReadWrite);

			/**
             * @brief Constructor. Creates a Texture2DArray from raw data. The data should be organized as a contiguous block of memory containing the pixel data for all slices in the texture array.
             * The width, height, depth, format, and access type must be specified to correctly interpret the raw data.
             * @param runtime The Runtime context in which the texture array will be created.
             * @param width The width of each texture slice in pixels.
             * @param height The height of each texture slice in pixels.
             * @param depth The number of texture slices in the array (i.e., the depth of the array).
             * @param data A pointer to the raw pixel data for the texture array. The data should be organized as a contiguous block of memory containing the pixel data for all slices in the texture array.
             * @param format The image format of the texture array (e.g., RGBA, etc.). See @ref ImageFormat for details.
             * @param type The access type for the texture array (e.g., ReadOnly, WriteOnly, ReadWrite).
             * @param access The access mode for the texture array (default is ReadWrite).
             */
			Texture2DArray(const Runtime& runtime, uint width, uint height, uint depth, void* data, ImageFormat format, AccessType type, AccessMode access = AccessMode::ReadWrite);

            /*
             * @brief Get the depth of the texture array, which is the number of texture slices in the array.
             * @return The depth of the texture array (i.e., the number of texture slices).
             */
			uint depth() const override;

			/*
			 * @brief Copy constructor
			 */
            Texture2DArray(const Texture2DArray& other);

            /*
             * @brief Copy assignment operator
             */
            Texture2DArray& operator=(const Texture2DArray& other);

            /**
             * @brief Fill a region of the texture array with a specified color. The region is defined by its starting coordinates (x, y, z) and its dimensions (w, h, d).
             * The color is specified as RGBA values (r, g, b, a).
             * @param r The red component of the fill color (0-255).
             * @param g The green component of the fill color (0-255).
             * @param b The blue component of the fill color (0-255).
             * @param a The alpha component of the fill color (0-255).
             * @param x The x-coordinate of the starting point of the region to fill.
             * @param y The y-coordinate of the starting point of the region to fill.
             * @param z The z-coordinate (depth) of the starting point of the region to fill.
             * @param w The width of the region to fill in pixels.
             * @param h The height of the region to fill in pixels.
             * @param d The depth of the region to fill (i.e., how many slices in the texture array to fill).
             */
			void fill_region(int r, int g, int b, int a, uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t h, uint32_t d);

            /**
             * @brief Retrieve a specific slice (depth) of the texture array as an OpenCV Mat object. The retrieved Mat will have the same width, height, and format as the texture array.
             * @param depth The index of the slice to retrieve (0-based). This should be less than the depth of the texture array.
             * @return An OpenCV Mat object containing the pixel data for the specified slice of the texture array.
             */
            cv::Mat retrieve_mat(uint depth) const;

            /**
             * @brief Write an OpenCV Mat object to a specific slice (depth) of the texture array. The Mat must have the same width, height, and format as the texture array.
             * @param depth The index of the slice to write to (0-based). This should be less than the depth of the texture array.
             * @param image An OpenCV Mat object containing the pixel data to write to the specified slice of the texture array. The Mat must have the same width, height, and format as the texture array.
             */
            void write_mat(uint depth, const cv::Mat& image);

    };

    typedef std::shared_ptr<Texture2DArray> RemoteTexture2DArrayPtr;
}


