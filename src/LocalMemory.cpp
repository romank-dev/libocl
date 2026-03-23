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

#include <libocl/LocalMemory.hpp>

#include "logging.hpp"

using namespace ocl;

LocalMemory::LocalMemory(uint size)
    : Argument(), _size(size)
{}

LocalMemory::LocalMemory(const LocalMemory& other)
    : Argument(other), _size(other._size)
{}

uint32_t LocalMemory::size() const
{
    return _size;
}

LocalMemory& LocalMemory::operator=(const LocalMemory& other)
{
    if(this != &other)
    {
        _size = other._size;
    }
    return *this;
}

void LocalMemory::bind_to_kernel(const Kernel& k, int idx) const
{
    CL_CHECK_RET(clSetKernelArg(k._kernel, idx, _size, nullptr));
}
