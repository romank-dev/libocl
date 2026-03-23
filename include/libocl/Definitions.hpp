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

namespace ocl
{

	enum class ImageFormat : int
	{
		Red, Alpha, // single channel simple layout (good for gray scale)

		Intensity, //  r=g=b=a (use only with CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT)
		Luminance, // r=g=b, a=1.0 (use only with CL_UNORM_INT8, CL_UNORM_INT16, CL_SNORM_INT8, CL_SNORM_INT16, CL_HALF_FLOAT, or CL_FLOAT)

		RGBA, // 4 channel classic (recommended)

		ARGB, BGRA, // can only be used with data type CL_UNORM_INT8, CL_SNORM_INT8, CL_SIGNED_INT8 or CL_UNSIGNED_INT8.

		RedGreen, RedAlpha, YUYV // 2 channels (why would you ever use this?)
	};

	enum class AccessType : int
	{
		Int8, Uint8, // byte per color
		Int8N, Uint8N, // normalized (use if you want to call read_imagef and work with cannonical float values)
		Int16, Uint16, // word per color
		Int16N, Uint16N, // normalized (use if you want to call read_imagef and work with cannonical float values)
		Int32, Uint32, // double word per color
		Float, // 32-bit float per color
		Half // 16-bit float per color (faster)
	};

    enum class AccessMode
    {
        ReadOnly,
        WriteOnly,
        ReadWrite
    };
}
