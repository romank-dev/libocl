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

#include <libocl/TextureBase.hpp>
#include <libocl/Definitions.hpp>
#include <libocl/Runtime.hpp>
#include <opencv2/opencv.hpp>
#include <memory>
#include <libocl/GlobalMemory.hpp>

namespace ocl
{

    class Texture2DArray;

    /**
     * @class Texture2D
     * @brief Represents a 2D texture in OpenCL. It is a more sophisticated memory type than RemoteBuffer, as it uses optimized texture memory.
     * When creating a Texture2D, specific image details such as width, height, format, and access type are provided.
     * Within an OpenCL kernel, a Texture2D is not accessed as a global pointer to memory but rather as a sampler object,
     * and you must use image-specific functions to work with textures.
     * Note that OpenCV support for Texture2D is minimal and straightforward, supporting only 1, 2, or 4 channels.
     */
    class Texture2D final : public ocl::TextureBase
    {
        public:
            /**
             * @brief Constructor. Creates a Texture2D with the specified width, height, format, access type, and access mode.
             * @param runtime The Runtime context in which the texture will be created.
             * @param width The width of the texture in pixels.
             * @param height The height of the texture in pixels.
             * @param format The image format of the texture (e.g., RGBA, etc.). See @ref ImageFormat for details.
             * @param type The access type for the texture (e.g., ReadOnly, WriteOnly, ReadWrite).
             * @param access The access mode for the texture (default is ReadWrite).
             */
            Texture2D(const Runtime& runtime, uint width, uint height, ImageFormat format, AccessType type, AccessMode access = AccessMode::ReadWrite);

            /**
             * @brief Constructor. Creates a Texture2D from an OpenCV Mat object. The data from the Mat will be uploaded to the texture.
             * The texture will have the same width, height, and format as the Mat.
             * @param runtime The Runtime context in which the texture will be created.
             * @param mat The OpenCV Mat object from which to create the texture. The texture will have the same width, height, and format as the Mat.
             * @param type The access type for the texture (e.g., ReadOnly, WriteOnly, ReadWrite).
             * @param access The access mode for the texture (default is ReadWrite).
             */
            Texture2D(const Runtime& runtime, const cv::Mat& mat, AccessType type, AccessMode access = AccessMode::ReadWrite);

            /**
             * @brief Constructor. Creates a Texture2D with the specified width, height, format, access type, and access mode, and initializes it with the provided data.
             * The data should be in a format compatible with the specified image format and access type.
             * @param runtime The Runtime context in which the texture will be created.
             * @param width The width of the texture in pixels.
             * @param height The height of the texture in pixels.
             * @param data A pointer to the data that will be uploaded to the texture. The data should be in a format compatible with the specified image format and access type.
             * @param format The image format of the texture (e.g., RGBA, etc.). See @ref ImageFormat for details.
             * @param type The access type for the texture (e.g., ReadOnly, WriteOnly, ReadWrite).
             * @param access The access mode for the texture (default is ReadWrite).
             */
            Texture2D(const Runtime& runtime, uint width, uint height, void* data, ImageFormat format, AccessType type, AccessMode access = AccessMode::ReadWrite);

            /**
             * @brief Copy constructor
             */
            Texture2D(const Texture2D& other);

            /**
             * @brief Copy assignment operator
             */
            Texture2D& operator=(const Texture2D& other);

            /**
             * @brief Copies the contents of this texture to another Texture2D
             * @param other The destination Texture2D to which the contents will be copied.
             *  The destination texture must have the same dimensions and format as this texture.
             */
            void copyTo(const Texture2D& other) const;

            /**
             * @brief Copies this texture to a specific depth slice (index) of a Texture2DArray
             * @param other The destination Texture2DArray to which the contents will be copied.
             * The destination texture array must have the same width, height, and format as this texture, and the specified depth must be within the bounds of the array.
             * @param depth The depth slice of this texture to copy (0-based index).
             */
            void copyTo(const Texture2DArray& other, uint32_t depth) const;

            /**
             * @brief Copies a specific region of this texture to a specific region of another Texture2D
             * @param other The destination Texture2D to which the contents will be copied.
             * The destination texture must have the same format as this texture, and the specified regions must be within the bounds of their respective textures.
             * @param fromX The x-coordinate of the top-left corner of the source region in this texture.
             * @param fromY The y-coordinate of the top-left corner of the source region in this texture.
             * @param toX The x-coordinate of the top-left corner of the destination region in the other texture.
             * @param toY The y-coordinate of the top-left corner of the destination region in the other texture.
             * @param w The width of the region to copy, in pixels.
             * @param h The height of the region to copy, in pixels.
             */
            void copyTo(const Texture2D& other, uint32_t fromX, uint32_t fromY, uint32_t toX, uint32_t toY, uint32_t w, uint32_t h) const;

            /**
             * @brief Returns the depth of the texture, which is always 1 for Texture2D.
             * @return The depth of the texture (always 1 for Texture2D).
             */
            uint depth() const override;

            /**
             * @brief Retrieves the contents of the texture as an OpenCV Mat object. The Mat will have the same width, height, and format as the texture.
             * @return An OpenCV Mat object containing the contents of the texture. The Mat will have the same width, height, and format as the texture.
             */
            cv::Mat retrieve_mat() const;

            /**
             * @brief Reads the contents of the texture into an existing OpenCV Mat object. The Mat must have the same width, height, and format as the texture.
             * @param image The OpenCV Mat object into which the contents of the texture will be read. The Mat must have the same width, height, and format as the texture.
             */
            void read_mat(cv::Mat& image) const;

            /**
             * @brief Reads a specific region of the texture into an existing OpenCV Mat object.
             * @param image The OpenCV Mat object into which the contents of the specified region of the texture will be read. The Mat must have the same width and height as the specified region, and the format must match the texture.
             * @param from_x The x-coordinate of the top-left corner of the source region in the texture.
             * @param from_y The y-coordinate of the top-left corner of the source region in the texture.
             * @param width The width of the region to read, in pixels.
             * @param height The height of the region to read, in pixels.
             */
            void read_region(cv::Mat& image, uint32_t from_x, uint32_t from_y, uint32_t width, uint32_t height) const;

            /**
             * @brief Writes the contents of an OpenCV Mat object to the texture. The Mat must have the same width, height, and format as the texture.
             * @param image The OpenCV Mat object containing the data to be written to the texture. The Mat must have the same width, height, and format as the texture.
             */
            void write_mat(const cv::Mat& image) const;

        private:

            int get_cv_type() const;

            static bool is_valid_coords(const Texture2D& texture, uint32_t x, uint32_t y);

    };

    typedef std::shared_ptr<Texture2D> RemoteTexture2DPtr;
}


