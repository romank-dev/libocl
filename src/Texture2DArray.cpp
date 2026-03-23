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

#include <libocl/Texture2DArray.hpp>
#include <libcommon/libcommon.hpp>
#include "logging.hpp"

using namespace ocl;
using namespace cv;

Texture2DArray::Texture2DArray(const Runtime& runtime, const Mat* mat, uint depth, AccessType type, AccessMode access)
    : Texture2DArray(runtime, mat->cols, mat->rows, depth, format_from_cv_mat(*mat), type, access)

{
	for(uint i = 0; i < depth; ++i)
	{
		size_t origin[] = {0,0,i};
		size_t region[] = { (size_t)mat->cols, (size_t)mat->rows, 1};
		CL_CHECK_RET(clEnqueueWriteImage(runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, mat[i].data, 0, NULL, NULL));
	}
}

Texture2DArray::Texture2DArray(const Runtime& runtime, uint width, uint height, uint depth, void* data, ImageFormat format, AccessType type, AccessMode access)
    : Texture2DArray(runtime, width, height, depth, format, type, access)
{
	size_t origin[] = {0,0,0};
	size_t region[] = {width, height, depth};
	CL_CHECK_RET(clEnqueueWriteImage(runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, data, 0, NULL, NULL));
}

Texture2DArray::Texture2DArray(const Runtime& runtime, uint width, uint height, uint depth, ImageFormat format, AccessType type, AccessMode access)
    : TextureBase(runtime, type, format, access)
{
    _image_format = format;
    _data_type = type;

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
    cl_image_desc desc
	{
    	CL_MEM_OBJECT_IMAGE2D_ARRAY,
		width,
		height,
		1,
		depth,
		0,
		0,
		0,
		0,
		nullptr
	};

    _tex = clCreateImage(_runtime._device._context, flags, &fmt, &desc, nullptr, &error_code);
    CL_CHECK_RET(error_code);
}

Texture2DArray::Texture2DArray(const Texture2DArray& other) :
	TextureBase(other)
{
	CL_CHECK_RET(clRetainMemObject(other._tex));
	_tex = other._tex;
}

Texture2DArray& Texture2DArray::operator =(const Texture2DArray& other)
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

Mat Texture2DArray::retrieve_mat(uint depth) const
{
	size_t w = width(), h = height();

	int type;
	if(_data_type <= AccessType::Uint8N)
	{
		if(_image_format == ImageFormat::BGRA || _image_format == ImageFormat::RGBA)
			type = CV_8UC4;
		else if(_image_format == ImageFormat::RedGreen || _image_format == ImageFormat::RedAlpha || _image_format == ImageFormat::YUYV)
			type = CV_8UC2;
		else if(_image_format == ImageFormat::Red || _image_format == ImageFormat::Alpha)
			type = CV_8UC1;
		else throw ::Exception("RemoteTexture2D::retrieve_mat(): Unsupported format for OpenCV!");
	}
	else if(_data_type <= AccessType::Uint16N)
	{
		if(_image_format == ImageFormat::BGRA || _image_format == ImageFormat::RGBA)
			type = CV_16UC4;
		else if(_image_format == ImageFormat::RedGreen || _image_format == ImageFormat::RedAlpha || _image_format == ImageFormat::YUYV)
			type = CV_16UC2;
		else if(_image_format == ImageFormat::Red || _image_format == ImageFormat::Alpha)
			type = CV_16UC1;
		else throw ::Exception("RemoteTexture2D::retrieve_mat(): Unsupported format for OpenCV!");
	}
	else if(_data_type == AccessType::Float)
	{
		if(_image_format == ImageFormat::BGRA || _image_format == ImageFormat::RGBA)
			type = CV_32FC4;
		else if(_image_format == ImageFormat::RedGreen || _image_format == ImageFormat::RedAlpha || _image_format == ImageFormat::YUYV)
			type = CV_32FC2;
		else if(_image_format == ImageFormat::Red || _image_format == ImageFormat::Alpha)
			type = CV_32FC1;
		else throw ::Exception("RemoteTexture2D::retrieve_mat(): Unsupported format for OpenCV!");
	}
	else if(_data_type == AccessType::Int32 || _data_type == AccessType::Uint32)
    {
        if(_image_format == ImageFormat::BGRA || _image_format == ImageFormat::RGBA)
            type = CV_32SC4;
        else if(_image_format == ImageFormat::RedGreen || _image_format == ImageFormat::RedAlpha || _image_format == ImageFormat::YUYV)
            type = CV_32SC2;
        else if(_image_format == ImageFormat::Red || _image_format == ImageFormat::Alpha)
            type = CV_32SC1;
        else throw ::Exception("RemoteTexture2D::retrieve_mat(): Unsupported format for OpenCV!");
    }
	else throw ::Exception("RemoteTexture2D::retrieve_mat(): Unsupported format for OpenCV!");

	Mat m(h, w, type);

	size_t origin[] = {0, 0, depth};
	size_t region[] = {w, h, 1};
	CHECK_THROW(depth < this->depth(), "Depth is incorrect! (d=%u)", this->depth());
	CL_CHECK_RET(clEnqueueReadImage(_runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, m.data, 0, NULL, NULL));

	return m;
}

uint Texture2DArray::depth() const
{
	size_t depth;
	CL_CHECK_RET(clGetImageInfo(_tex, CL_IMAGE_ARRAY_SIZE, sizeof(depth), &depth, nullptr));
	return depth;
}

void Texture2DArray::write_mat(uint depth, const Mat& image)
{
	CHECK_THROW(format_from_cv_mat(image) == _image_format, "Incorrect format!");

	size_t origin[] = {0,0,depth};
	size_t region[] = {width(), height(), 1};
	CHECK_THROW((uint)region[0] == image.cols && (uint)region[1] == image.rows, "Image size is incorrect!");
	CHECK_THROW((uint)region[2] < this->depth(), "Depth is incorrect! (d=%u)", this->depth());

	CL_CHECK_RET(clEnqueueWriteImage(_runtime._command_queue, _tex, CL_TRUE, origin, region, 0, 0, image.data, 0, NULL, NULL));
}


void Texture2DArray::fill_region(int r, int g, int b, int a, uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t h, uint32_t d)
{
	int color[4] = { r,g,b,a };
	size_t origin[3] = { x,y,z };
	size_t region[3] = { w,h,d };
	CL_CHECK_RET(clEnqueueFillImage(_runtime._command_queue, _tex, color, origin, region, 0, nullptr, nullptr));
}

