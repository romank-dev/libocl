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
#include <libocl/GlobalMemory.hpp>

namespace ocl
{

    class Texture2DArray;

	/**
	 * A more sophisticated memory than RemoteBuffer since it uses the optimized texture memory.
	 * Specific image details are given when creating a texture. Within a kernel it is not a global
	 * pointer to memory but a sampler object and you must use image-specific functions to work with textures.
	 * NOTE: OpenCV support is very minimal and straight-forward (1,2,4 channels only are supported)
	 */
    class Texture2D final : public ocl::TextureBase
    {
        public:

    		~Texture2D() = default;

            Texture2D(const Runtime& runtime, uint width, uint height, ImageFormat format, AccessType type, AccessMode access = AccessMode::ReadWrite);
            Texture2D(const Runtime& runtime, const cv::Mat& mat, AccessType type, AccessMode access = AccessMode::ReadWrite);
            Texture2D(const Runtime& runtime, uint width, uint height, void* data, ImageFormat format, AccessType type, AccessMode access = AccessMode::ReadWrite);

            Texture2D(const Texture2D& other);
            Texture2D& operator=(const Texture2D& other);

            void copyTo(const Texture2D& other) const;
            void copyTo(const Texture2DArray& other, uint32_t depth) const;
            void copyTo(const Texture2D& other, uint32_t fromX, uint32_t fromY, uint32_t toX, uint32_t toY, uint32_t w, uint32_t h) const;

            uint depth() const override;

            // opencv integration
            cv::Mat retrieve_mat() const;
            void read_mat(cv::Mat& image) const;
            void read_region(cv::Mat& image, uint32_t from_x, uint32_t from_y, uint32_t width, uint32_t height) const;
            void write_mat(const cv::Mat& image) const;

        private:

            int get_cv_type() const;

            static bool is_valid_coords(const Texture2D& texture, uint32_t x, uint32_t y);

    };

    typedef std::shared_ptr<Texture2D> RemoteTexture2DPtr;
}


