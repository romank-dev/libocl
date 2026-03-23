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
	 * A more sophisticated memory than RemoteBuffer since it uses the optimized texture memory.
	 * Specific image details are given when creating a texture. Within a kernel it is not a global
	 * pointer to memory but a sampler object and you must use image-specific functions to work with textures.
	 * NOTE: OpenCV support is very minimal and straight-forward (1,2,4 channels only are supported)
	 */
    class Texture2DArray final : public ocl::TextureBase
    {
        friend class Texture2D;
        public:

			Texture2DArray(const Runtime& runtime, uint width, uint height, uint depth, ImageFormat format, AccessType type, AccessMode access = AccessMode::ReadWrite);
			Texture2DArray(const Runtime& runtime, const cv::Mat* mat_array, uint depth, AccessType type, AccessMode access = AccessMode::ReadWrite);
			Texture2DArray(const Runtime& runtime, uint width, uint height, uint depth, void* data, ImageFormat format, AccessType type, AccessMode access = AccessMode::ReadWrite);

			~Texture2DArray() = default;

			uint depth() const override;

            Texture2DArray(const Texture2DArray& other);
            Texture2DArray& operator=(const Texture2DArray& other);

			void fill_region(int r, int g, int b, int a, uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t h, uint32_t d);

            // opencv integration
            cv::Mat retrieve_mat(uint depth) const;
            void write_mat(uint depth, const cv::Mat& image);

    };

    typedef std::shared_ptr<Texture2DArray> RemoteTexture2DArrayPtr;
}


