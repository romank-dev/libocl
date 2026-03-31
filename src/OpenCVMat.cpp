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


#include <libocl/OpenCVMat.hpp>
#include "logging.hpp"

using namespace ocl;
using namespace cv;

OpenCVMat::OpenCVMat(const Runtime& runtime, const Mat& image, AccessMode access_mode)
    : GlobalMemory(runtime, image.data, image.rows*image.step, access_mode)
{
    _cv_size = image.size();
    _cv_type = image.type();
}

OpenCVMat::OpenCVMat(const Runtime& runtime, Size cv_size, int cv_type, AccessMode access_mode)
    : GlobalMemory(runtime, cv_size.height*cv_size.width*type_to_elemsize(cv_type), access_mode)
{
    _cv_size = cv_size;
    _cv_type = cv_type;
}

OpenCVMat::OpenCVMat(const OpenCVMat& other)
    : GlobalMemory(other)
{
    _cv_size = other._cv_size;
    _cv_type = other._cv_type;
}

OpenCVMat& OpenCVMat::operator=(const OpenCVMat& other)
{
    if(this != &other)
    {
        _cv_size = other._cv_size;
        _cv_type = other._cv_type;
    }
    return *this;
}

Size OpenCVMat::cv_size() const
{
    return _cv_size;
}

int OpenCVMat::cv_type() const
{
    return _cv_type;
}

int OpenCVMat::type_to_elemsize(int cv_type)
{
    auto mat = Mat(0,0,cv_type);
    return mat.elemSize();
}

Mat OpenCVMat::retrieve() const
{
    Mat image(_cv_size, _cv_type);
    CL_CHECK_RET(clEnqueueReadBuffer(_runtime._command_queue, _buffer, CL_TRUE, 0, size(), (void*)image.data, 0, NULL, NULL));
    return image;
}

void OpenCVMat::retrieve(Mat& image) const
{
    CL_CHECK_RET(clEnqueueReadBuffer(_runtime._command_queue, _buffer, CL_TRUE, 0, size(), (void*)image.data, 0, NULL, NULL));
}


void OpenCVMat::write_opencvcv_mat(const Mat& image)
{
    CHECK_THROW(image.type() == _cv_type, "RemoteOpenCVMat::write_OpenCVMat requires the same image type");
    CHECK_THROW(image.size() == _cv_size, "RemoteOpenCVMat::write_OpenCVMat requires the same image size");

    CL_CHECK_RET(clEnqueueWriteBuffer(_runtime._command_queue, _buffer, CL_TRUE, 0, size(), image.data, 0, NULL, NULL));
}

