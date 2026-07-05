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
     * @class OpenCVMat
     * @brief This class represents an OpenCV Mat object in raw form, hence inherits GlobalMemory.
     * It allows you to easily translate between OpenCV Mat objects and OpenCL buffers, making it convenient to use OpenCV images as input or output for OpenCL kernels.
     * NOTE: The data in the OpenCL buffer is stored in the same format and layout as the OpenCV Mat.
     */
    class OpenCVMat final : public ocl::GlobalMemory
    {
        public:

            /**
             * @brief Constructor. Creates an on-device OpenCV Mat object and copies a given OpenCV Mat into it.
             * @param runtime The Runtime object that manages the OpenCL context and command queue.
             * @param image The OpenCV Mat object to be represented as an OpenCL buffer.
             * @param access The access mode for the OpenCL buffer (read, write, or read/write). Default is read/write.
             */
            OpenCVMat(const Runtime& runtime, const cv::Mat& image, AccessMode access = AccessMode::ReadWrite);

            /**
             * @brief Constructor. Creates an on-device OpenCV Mat object with it's data being uninitialized.
             * @param runtime The Runtime object that manages the OpenCL context and command queue.
             * @param cv_size The size of the OpenCV Mat (width and height).
             * @param cv_type The type of the OpenCV Mat (e.g., CV_8UC3 for an 8-bit unsigned 3-channel image).
             * @param access The access mode for the OpenCL buffer (read, write, or read/write). Default is read/write.
             */
            OpenCVMat(const Runtime& runtime, cv::Size cv_size, int cv_type, AccessMode access = AccessMode::ReadWrite);

            /**
             * Copy constructor
             */
            OpenCVMat(const OpenCVMat& other);

            /**
             * Copy assignment operator
             */
            OpenCVMat& operator=(const OpenCVMat& other);

            /**
             * @brief Retrieve the data from the OpenCL buffer and return it as an OpenCV Mat object. The data is copied from the device to the host.
             * @return An OpenCV Mat object containing the data retrieved from the OpenCL buffer.
             */
            cv::Mat retrieve() const;

            /**
             * @brief Retrieve the data from the OpenCL buffer and copy it into a given OpenCV Mat object. The data is copied from the device to the host.
             * This variant of the retrieve function allows you to specify an existing OpenCV Mat object where the memory is pre-allocated.
             * @param mat The OpenCV Mat object where the retrieved data will be copied. It should have the same size and type as the original Mat used to create this OpenCVMat.
             */
            void retrieve(cv::Mat&) const;

            /**
             * @brief Write data from a given OpenCV Mat object into the OpenCL buffer. The data is copied from the host to the device.
             * @param image The OpenCV Mat object containing the data to be written into the OpenCL buffer. It should have the same size and type as the original Mat used to create this OpenCVMat.
             */
            void write_opencvcv_mat(const cv::Mat& image);

            /**
             * @brief Get the size of the OpenCV Mat (width and height).
             * @return The size of the OpenCV Mat as a cv::Size object, where width is the number of columns and height is the number of rows.
             */
            cv::Size cv_size() const;

            /**
             * @brief Get the type of the OpenCV Mat (e.g., CV_8UC3 for an 8-bit unsigned 3-channel image).
             * @return The type of the OpenCV Mat as an integer, which corresponds to the OpenCV type constants (e.g., CV_8UC3).
             */
            int cv_type() const;

        private:

            static int type_to_elemsize(int cv_type);

            int         _cv_type;
            cv::Size    _cv_size;
    };

}



