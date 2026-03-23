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

#include <libocl/TextureBase.hpp>
#include <map>
#include <libcommon/libcommon.hpp>
#include "logging.hpp"

using namespace ocl;
using namespace cv;

TextureBase::TextureBase(const Runtime& runtime, AccessType type, ImageFormat format, AccessMode access)
	: _data_type(type), _image_format(format), _runtime(runtime), _tex(0)
{}

TextureBase::TextureBase(const TextureBase& other) :
	_data_type(other._data_type),
	_image_format(other._image_format),
	_runtime(other._runtime)
{
	CL_CHECK_RET(clRetainMemObject(other._tex));
	_tex = other._tex;
}

TextureBase& TextureBase::operator =(const TextureBase& other)
{
	if(this != &other)
	{
		_data_type = other._data_type;
		_image_format = other._image_format;

		CL_CHECK_RET(clReleaseMemObject(_tex));
		CL_CHECK_RET(clRetainMemObject(other._tex));
		_tex = other._tex;
	}
	return *this;
}


TextureBase::~TextureBase()
{
	SAFE_DESTRUCTOR
	(
		CL_CHECK_RET(clReleaseMemObject(_tex));
	)
}

uint32_t TextureBase::size() const
{
    size_t res;
    CL_CHECK_RET(clGetMemObjectInfo(_tex, CL_MEM_SIZE, sizeof(res), &res, nullptr));
    return res;
}

ImageFormat TextureBase::format_from_cv_mat(const Mat& mat)
{
    switch(mat.channels())
    {
        case 4:
            return ImageFormat::RGBA; // openCV uses BGRA for what ever the fuck reason
        case 2:
            return ImageFormat::RedGreen;
        case 1:
            return ImageFormat::Red;
        default:
            throw ::Exception("RemoteTexture2D::format_from_cv_mat: Unsupported image format!");
    }
    return ImageFormat::ARGB;
}

cl_image_format TextureBase::create_format(ImageFormat format, AccessType type)
{
    const std::map<ImageFormat, cl_channel_order> image_types =
    {
        {ImageFormat::Red, CL_R},
        {ImageFormat::Alpha, CL_A},
        {ImageFormat::Intensity, CL_INTENSITY},
        {ImageFormat::Luminance, CL_LUMINANCE},
        {ImageFormat::RGBA, CL_RGBA},
        {ImageFormat::ARGB, CL_ARGB},
        {ImageFormat::BGRA, CL_BGRA},
        {ImageFormat::RedGreen, CL_RG},
        {ImageFormat::RedAlpha, CL_RA},
        {ImageFormat::YUYV, CL_RG}
    };

    const std::map<AccessType, cl_channel_type> data_types =
    {
        {AccessType::Int8, CL_SIGNED_INT8},
        {AccessType::Uint8, CL_UNSIGNED_INT8},
        {AccessType::Int8N, CL_SNORM_INT8},
        {AccessType::Uint8N, CL_UNORM_INT8},
        {AccessType::Int16, CL_SIGNED_INT16},
        {AccessType::Uint16, CL_UNSIGNED_INT16},
        {AccessType::Int16N, CL_SNORM_INT16},
        {AccessType::Uint16N, CL_UNORM_INT16},
        {AccessType::Int32, CL_SIGNED_INT32},
        {AccessType::Uint32, CL_UNSIGNED_INT32},
        {AccessType::Float, CL_FLOAT},
        {AccessType::Half, CL_HALF_FLOAT},
    };

    CHECK_THROW(image_types.find(format) != image_types.end(), "RemoteTexture2D::create_format: Unsupported format!");
    CHECK_THROW(data_types.find(type) != data_types.end(), "RemoteTexture2D::create_format: Unsupported data type!");

    cl_image_format fmt =
    {
        image_types.find(format)->second,   //image_channel_order
        data_types.find(type)->second       //image_channel_data_type
    };

    return fmt;
}

void TextureBase::bind_to_kernel(const Kernel& k, int idx) const
{
	CL_CHECK_RET(clSetKernelArg(k._kernel, idx, sizeof(cl_mem), (void*)&_tex));
}


uint TextureBase::width() const
{
	size_t width;
	CL_CHECK_RET(clGetImageInfo(_tex, CL_IMAGE_WIDTH, sizeof(width), &width, nullptr));
	return width;
}

uint TextureBase::height() const
{
	size_t height;
	CL_CHECK_RET(clGetImageInfo(_tex, CL_IMAGE_HEIGHT, sizeof(height), &height, nullptr));
	return height;
}

ImageFormat TextureBase::format() const
{
	return _image_format;
}

AccessType TextureBase::type() const
{
	return _data_type;
}


void TextureBase::retrieve(void* data) const
{
	size_t origin[] = {0,0,0};
	size_t region[] = {width(), height(), depth()};
	CL_CHECK_RET(clEnqueueReadImage(_runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL));
}

void TextureBase::write(void* data)
{
	size_t origin[] = {0,0,0};
	size_t region[] = {width(), height(), depth()};
	CL_CHECK_RET(clEnqueueWriteImage(_runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL));
}

void TextureBase::fill(int r, int g, int b, int a)
{
	int color[4] = { r,g,b,a };
	size_t origin[3] = { 0 };
	size_t region[3] = { width(), height(), depth() };
	CL_CHECK_RET(clEnqueueFillImage(_runtime._command_queue, _tex, color, origin, region, 0, nullptr, nullptr));
}

