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

#include <libocl/GlobalMemory.hpp>
#include "logging.hpp"

namespace ocl
{
    GlobalMemory::GlobalMemory(const Runtime& runtime, uint size, AccessMode access)
        : _runtime(runtime)
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

        cl_int error_code;
        _buffer = clCreateBuffer(runtime._device._context, flags, size, NULL, &error_code);
        CL_CHECK_RET(error_code);
    }

    GlobalMemory::GlobalMemory(const Runtime& runtime, void* data, uint size, AccessMode access)
        : GlobalMemory(runtime, size, access)
    {
        ExceptionGuard cleaner([&]()
        {
            this->~GlobalMemory();
        });
        CL_CHECK_RET(clEnqueueWriteBuffer(runtime._command_queue, _buffer, CL_TRUE, 0, size, data, 0, NULL, NULL));
    }

    GlobalMemory::GlobalMemory(const GlobalMemory& other)
        : _runtime(other._runtime)
    {
        CL_CHECK_RET(clRetainMemObject(other._buffer));
        _buffer = other._buffer;
    }

    uint32_t GlobalMemory::size() const
    {
        size_t res;
        CL_CHECK_RET(clGetMemObjectInfo(_buffer, CL_MEM_SIZE, sizeof(res), &res, nullptr));
        return res;
    }

    GlobalMemory& GlobalMemory::operator=(const GlobalMemory& other)
    {
        if(this != &other)
        {
            CL_CHECK_RET(clReleaseMemObject(_buffer));
            CL_CHECK_RET(clRetainMemObject(other._buffer));
            _buffer = other._buffer;
        }
        return *this;
    }

    GlobalMemory::~GlobalMemory()
    {
        SAFE_DESTRUCTOR
        (
            CL_CHECK_RET(clReleaseMemObject(_buffer));
        )
    }

    void GlobalMemory::retrieve(void* data, uint offset, uint size) const
    {
        CL_CHECK_RET(clEnqueueReadBuffer(_runtime._command_queue, _buffer, CL_TRUE, offset, size, data, 0, nullptr, nullptr));
    }

    void GlobalMemory::bind_to_kernel(const Kernel& k, int idx) const
    {
        CL_CHECK_RET(clSetKernelArg(k._kernel, idx, sizeof(cl_mem), (void*)&_buffer));
    }

    GlobalMemory::GlobalMemory(const Runtime& runtime, cl_mem buffer, uint size)
        : _runtime(runtime)
    {
        _buffer = buffer;
    }

    void GlobalMemory::copy_to(const GlobalMemory& other) const
    {
        CL_CHECK_RET(clEnqueueCopyBuffer(_runtime._command_queue, _buffer, other._buffer, 0, 0, size(), 0, nullptr, nullptr));
    }

    void GlobalMemory::copy_to(const GlobalMemory& other, uint32_t src_offset, uint32_t dst_offset, uint32_t size) const
    {
        CL_CHECK_RET(clEnqueueCopyBuffer(_runtime._command_queue, _buffer, other._buffer, src_offset, dst_offset, size, 0, nullptr, nullptr));
    }

    void GlobalMemory::fill( void* pattern, uint32_t pattern_size, uint32_t offset, uint32_t size) const
    {
        CL_CHECK_RET(clEnqueueFillBuffer(_runtime._command_queue, _buffer, pattern, pattern_size, offset, size, 0, nullptr, nullptr));
    }

}







