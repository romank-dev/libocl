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
	 * A more sophisticated memory than RemoteBuffer since it uses the optimized texture memory.
	 * Specific image details are given when creating a texture. Within a kernel it is not a global
	 * pointer to memory but a sampler object and you must use image-specific functions to work with textures.
	 * NOTE: OpenCV support is very minimal and straight-forward (1,2,4 channels only are supported)
	 */
    class TextureBase : public ocl::Argument
    {
        public:
    	    virtual ~TextureBase();

            // raw read/write
            void retrieve(void* data) const;
            void write(void* data);

            // getters
            uint width() const;
            uint height() const;
            virtual uint depth() const = 0;
            ImageFormat format() const;
            AccessType type() const;
            virtual uint32_t size() const override;

            void fill(int r, int g, int b, int a);

            TextureBase(const TextureBase& other);
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


