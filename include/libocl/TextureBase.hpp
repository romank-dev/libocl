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
#include <libocl/GlobalMemory.hpp>

namespace ocl
{
    /**
     * @class TextureBase
     * @brief Base class for texture objects in OpenCL. This class provides common functionality for different types of textures (e.g., Texture2D, Texture2DArray).
     * It manages the underlying OpenCL image object and provides methods for retrieving and writing raw data, as well as getting texture properties such as width, height, format, and access type.
     * Derived classes must implement the depth() method to specify the depth of the texture (e.g., 1 for Texture2D, or the number of slices for Texture2DArray).
     */
    class TextureBase : public ocl::Argument
    {
        public:
            /**
             * Virtual destructor
             */
    	    virtual ~TextureBase();

            /**
             * @brief Retrieve the raw pixel data from the texture.
             * The data will be copied into the provided pointer, which should point to a block of memory large enough to hold the texture data.
             * The format and organization of the data will depend on the texture's properties (e.g., width, height, format, access type).
             * @param data A pointer to a block of memory where the texture data will be copied.
             * The caller is responsible for ensuring that this memory is large enough to hold the texture data.
             */
            void retrieve(void* data) const;

            /**
             * @brief Write raw pixel data to the texture.
             * The data should be organized in a format compatible with the texture's properties (e.g., width, height, format, access type).
             * The provided pointer should point to a block of memory containing the pixel data to be written to the texture.
             * @param data A pointer to a block of memory containing the pixel data to write to the texture.
             * The data should be organized in a format compatible with the texture's properties (e.g., width, height, format, access type).
             */
            void write(void* data);

            /**
             * @brief Get the width of the texture in pixels.
             * @return The width of the texture in pixels.
             */
            uint width() const;

            /**
             * @brief Get the height of the texture in pixels.
             * @return The height of the texture in pixels.
             */
            uint height() const;

            /**
             * @brief Get the depth of the texture. For a Texture2D, this will return 1. For a Texture2DArray, this will return the number of slices in the array.
             * @return The depth of the texture (e.g., 1 for Texture2D, or the number of slices for Texture2DArray).
             */
            virtual uint depth() const = 0;

            /**
             * @brief Get the image format of the texture (e.g., RGBA, etc.). See @ref ImageFormat for details.
             * @return The image format of the texture.
             */
            ImageFormat format() const;

            /**
             * @brief Get the access type for the texture (e.g., ReadOnly, WriteOnly, ReadWrite).
             * @return The access type for the texture.
             */
            AccessType type() const;

            /**
             * @brief Get the size of the texture in bytes. This is the total size of the texture data,
             * which can be calculated as width * height * depth * bytes_per_pixel (where bytes_per_pixel depends on the image format and access type).
             * @return The size of the texture in bytes.
             */
            virtual uint32_t size() const override;

            /**
             * @brief Fill the texture with a solid color specified by the RGBA components. Each component should be in the range [0, 255].
             * This method will fill the entire texture with the specified color.
             * @param r The red component of the color (0-255).
             * @param g The green component of the color (0-255).
             * @param b The blue component of the color (0-255).
             * @param a The alpha component of the color (0-255).
             */
            void fill(int r, int g, int b, int a);

            /**
             * @brief Copy constructor
             */
            TextureBase(const TextureBase& other);

            /**
             * @brief Copy assignment operator
             */
			TextureBase& operator=(const TextureBase& other);

        protected:
            TextureBase(const Runtime& runtime, AccessType type, ImageFormat format, AccessMode access = AccessMode::ReadWrite);

            static ImageFormat format_from_cv_mat(const cv::Mat& mat);
            static cl_image_format create_format(ImageFormat format, AccessType type);

            virtual void bind_to_kernel(const Kernel& k, int idx) const;

        protected:
            AccessType _data_type;
            ImageFormat _image_format;
            Runtime _runtime;
            cl_mem _tex;
    };
}


