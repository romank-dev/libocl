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

#include <opencv2/opencv.hpp>
#include <libocl/GlobalMemory.hpp>

namespace ocl
{
	/**
	 * Represents an openCV Mat object in raw form, hence inherits RemoteBuffer.
	 * Within a kernel it will appear as a __global array, and accessed like normal memory.
	 */
    class OpenCVMat final : public ocl::GlobalMemory
    {
        public:

            OpenCVMat(const Runtime& runtime, const cv::Mat& image, AccessMode access = AccessMode::ReadWrite);
            OpenCVMat(const Runtime& runtime, void* data, cv::Size cv_size, int cv_type, AccessMode access = AccessMode::ReadWrite);
            OpenCVMat(const Runtime& runtime, cv::Size cv_size, int cv_type, AccessMode access = AccessMode::ReadWrite);

            virtual ~OpenCVMat();

            OpenCVMat(const OpenCVMat& other);
            OpenCVMat& operator=(const OpenCVMat& other);

            cv::Mat retrieve() const;
            void retrieve(cv::Mat&) const;
            void write_opencvcv_mat(const cv::Mat& image);

            cv::Size cv_size() const;
            int cv_type() const;

        private:

            static int type_to_elemsize(int cv_type);

            int         _cv_type;
            cv::Size    _cv_size;
    };

    typedef std::shared_ptr<OpenCVMat> RemoteMatPtr;

}



