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
#include <libocl/Runtime.hpp>
#include <libocl/CLException.hpp>
#include "logging.hpp"

using namespace ocl;

Runtime::Runtime(const Device& device)
    : _device(device)
{
    cl_int error_code;
    _command_queue = clCreateCommandQueue(device._context, device._device, 0, &error_code);
    CL_CHECK_RET(error_code);
}

Runtime::~Runtime()
{
    SAFE_DESTRUCTOR
    (
        CL_CHECK_RET(clReleaseCommandQueue(_command_queue));
    );
}

Runtime::Runtime(const Runtime& other)
    : _device(other._device), _command_queue(other._command_queue)
{
    CL_CHECK_RET(clRetainCommandQueue(_command_queue));
}

Runtime& Runtime::operator=(const Runtime& other)
{
    if(this != &other)
    {
        CL_CHECK_RET(clReleaseCommandQueue(_command_queue));
        _command_queue = other._command_queue;
        _device = other._device;
        CL_CHECK_RET(clRetainCommandQueue(_command_queue));
    }
    return *this;
}

Device& Runtime::device()
{
	return _device;
}

const Device& Runtime::device() const
{
	return _device;
}

void Runtime::wait_for_task_completion()
{
    CL_CHECK_RET(clFlush(_command_queue));
    CL_CHECK_RET(clFinish(_command_queue));
}



