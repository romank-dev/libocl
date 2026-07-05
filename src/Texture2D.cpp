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

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <libocl/Texture2D.hpp>
#include <libocl/Texture2DArray.hpp>
#include <map>
#include <libcommon/libcommon.hpp>
#include "logging.hpp"

using namespace ocl;
using namespace cv;

Texture2D::Texture2D(const Runtime& runtime, const Mat& mat, AccessType type, AccessMode access)
: Texture2D(runtime, mat.cols, mat.rows, format_from_cv_mat(mat), type, access)

{
    size_t origin[] = {0,0,0};
    size_t region[] = { (size_t)mat.cols, (size_t)mat.rows, 1 };
    CL_CHECK_RET(clEnqueueWriteImage(runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, mat.data, 0, NULL, NULL));
}

Texture2D::Texture2D(const Runtime& runtime, uint width, uint height, void* data, ImageFormat format, AccessType type, AccessMode access)
: Texture2D(runtime, width, height, format, type, access)
{
    size_t origin[] = {0,0,0};
    size_t region[] = {width, height, 1};
    CL_CHECK_RET(clEnqueueWriteImage(runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL));
}

Texture2D::Texture2D(const Runtime& runtime, uint width, uint height, ImageFormat format, AccessType type, AccessMode access)
: TextureBase(runtime, type, format, access)
{
    ulong flags;
    switch(access)
    {
        case AccessMode::ReadOnly:
            flags = CL_MEM_READ_ONLY;
            break;
        case AccessMode::WriteOnly:
            flags = CL_MEM_WRITE_ONLY;
            break;
        case AccessMode::ReadWrite:
            flags = CL_MEM_READ_WRITE;
            break;
    };

    cl_image_format fmt = create_format(_image_format, type);
    cl_int error_code;
    _tex = clCreateImage2D(_runtime._device._context, flags, &fmt, width, height, 0, NULL, &error_code);
    CL_CHECK_RET(error_code);
}

Texture2D::Texture2D(const Texture2D& other) :
	        TextureBase(other)
{}

Texture2D& Texture2D::operator =(const Texture2D& other)
{
    TextureBase::operator =(other);
    return *this;
}

void Texture2D::copyTo(const Texture2D& other) const
{
    CHECK_THROW(other.width() == width() && other.height() == height() && other.type() == type(), "Texture dimensions must match!");
    const size_t src_origin[] = { 0, 0, 0 };
    const size_t dst_origin[] = { 0, 0, 0 };
    const size_t region[] = { width(), height(), 1 };
    CL_CHECK_RET(clEnqueueCopyImage(_runtime._command_queue, _tex, other._tex, src_origin, dst_origin, region, 0, nullptr, nullptr));
}

void Texture2D::copyTo(const Texture2DArray& other, uint32_t depth) const
{
    CHECK_THROW(other.width() == width() && other.height() == height() && other.type() == type(), "Texture dimensions must match!");
    const size_t src_origin[] = { 0, 0, 0 };
    const size_t dst_origin[] = { 0, 0, depth };
    const size_t region[] = { width(), height(), 1 };
    CL_CHECK_RET(clEnqueueCopyImage(_runtime._command_queue, _tex, other._tex, src_origin, dst_origin, region, 0, nullptr, nullptr));
}

bool Texture2D::is_valid_coords(const Texture2D& texture, uint32_t x, uint32_t y)
{
    return x < texture.width() && y < texture.height();
}

void Texture2D::copyTo(const Texture2D& other, uint32_t fromX, uint32_t fromY, uint32_t toX, uint32_t toY, uint32_t w, uint32_t h) const
{
    CHECK_THROW(is_valid_coords(*this, fromX, fromY), "Source coords are out of bounds!");
    CHECK_THROW(is_valid_coords(other, toX, toY), "Destination coords are out of bounds!");
    CHECK_THROW(is_valid_coords(other, toX+w-1, toY+h-1), "Destination coords+range are out of bounds!");
    CHECK_THROW(is_valid_coords(*this, fromX+w-1, fromY+h-1), "Source coords+range are out of bounds!");
    const size_t src_origin[] = { fromX, fromY, 0 };
    const size_t dst_origin[] = { toX, toY, 0 };
    const size_t region[] = { w, h, 1 };
    CL_CHECK_RET(clEnqueueCopyImage(_runtime._command_queue, _tex, other._tex, src_origin, dst_origin, region, 0, nullptr, nullptr));
}

int Texture2D::get_cv_type() const
{
    int type;
    if(_data_type <= AccessType::Uint8N)
    {
        if(_image_format == ImageFormat::BGRA || _image_format == ImageFormat::RGBA)
            type = CV_8UC4;
        else if(_image_format == ImageFormat::RedGreen || _image_format == ImageFormat::RedAlpha || _image_format == ImageFormat::YUYV)
            type = CV_8UC2;
        else if(_image_format == ImageFormat::Red || _image_format == ImageFormat::Alpha)
            type = CV_8UC1;
        else throw ::Exception("Texture2D::retrieve_mat(): Unsupported format for OpenCV!");
    }
    else if(_data_type <= AccessType::Uint16N)
    {
        if(_image_format == ImageFormat::BGRA || _image_format == ImageFormat::RGBA)
            type = CV_16UC4;
        else if(_image_format == ImageFormat::RedGreen || _image_format == ImageFormat::RedAlpha || _image_format == ImageFormat::YUYV)
            type = CV_16UC2;
        else if(_image_format == ImageFormat::Red || _image_format == ImageFormat::Alpha)
            type = CV_16UC1;
        else throw ::Exception("Texture2D::retrieve_mat(): Unsupported format for OpenCV!");
    }
    else if(_data_type == AccessType::Float)
    {
        if(_image_format == ImageFormat::BGRA || _image_format == ImageFormat::RGBA)
            type = CV_32FC4;
        else if(_image_format == ImageFormat::RedGreen || _image_format == ImageFormat::RedAlpha || _image_format == ImageFormat::YUYV)
            type = CV_32FC2;
        else if(_image_format == ImageFormat::Red || _image_format == ImageFormat::Alpha)
            type = CV_32FC1;
        else throw ::Exception("Texture2D::retrieve_mat(): Unsupported format for OpenCV!");
    }
    else if(_data_type == AccessType::Int32 || _data_type == AccessType::Uint32)
    {
        if(_image_format == ImageFormat::BGRA || _image_format == ImageFormat::RGBA)
            type = CV_32SC4;
        else if(_image_format == ImageFormat::RedGreen || _image_format == ImageFormat::RedAlpha || _image_format == ImageFormat::YUYV)
            type = CV_32SC2;
        else if(_image_format == ImageFormat::Red || _image_format == ImageFormat::Alpha)
            type = CV_32SC1;
        else throw ::Exception("Texture2D::retrieve_mat(): Unsupported format for OpenCV!");
    }
    else throw ::Exception("Texture2D::retrieve_mat(): Unsupported format for OpenCV!");
    return type;
}

void Texture2D::read_mat(Mat& image) const
{
    size_t w = width(), h = height();
    int type = get_cv_type();
    CHECK_THROW(w == image.cols && h == image.rows && type == image.type(), "OpenCV image is incompatible with texture");

    size_t origin[] = {0,0,0};
    size_t region[] = {w, h, 1};
    CL_CHECK_RET(clEnqueueReadImage(_runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, image.data, 0, NULL, NULL));
}

void Texture2D::read_region(Mat& image, uint32_t from_x, uint32_t from_y, uint32_t width, uint32_t height) const
{
    CHECK_THROW(image.cols == width && image.rows == height, "Image dimensions must match the given ROI in texture");
    size_t origin[] = {from_x, from_y, 0};
    size_t region[] = {width, height, 1};
    CL_CHECK_RET(clEnqueueReadImage(_runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, image.data, 0, NULL, NULL));
}

Mat Texture2D::retrieve_mat() const
{
    size_t w = width(), h = height();

    int type = get_cv_type();

    Mat m(h, w, type);

    size_t origin[] = {0,0,0};
    size_t region[] = {w, h, 1};
    CL_CHECK_RET(clEnqueueReadImage(_runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, m.data, 0, NULL, NULL));

    return m;
}

void Texture2D::write_mat(const Mat& image) const
{
    CHECK_THROW(format_from_cv_mat(image) == _image_format ||
    		(format_from_cv_mat(image) == ImageFormat::BGRA && _image_format == ImageFormat::RGBA) ||
    		(format_from_cv_mat(image) == ImageFormat::RGBA && _image_format == ImageFormat::BGRA)
			, "Incorrect format!");

    size_t origin[] = {0,0,0};
    size_t region[] = {width(), height(), 1};
    CHECK_THROW((uint)region[0] == image.cols && (uint)region[1] == image.rows, "Image size is incorrect! (%dx%d)!=(%dx%d)",width(),height(),image.cols,image.rows);

    CL_CHECK_RET(clEnqueueWriteImage(_runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, image.data, 0, NULL, NULL));
}

uint Texture2D::depth() const
{
	return 1;
}



