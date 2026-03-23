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


#include <libocl/Runtime.hpp>
#include <libocl/Argument.hpp>

namespace ocl
{
    class GlobalMemory : public Argument
    {
        public:

            GlobalMemory(const Runtime& runtime, uint size, AccessMode access = AccessMode::ReadWrite);
            GlobalMemory(const Runtime& runtime, void* data, uint size, AccessMode access = AccessMode::ReadWrite);

            GlobalMemory(const GlobalMemory& other);
            GlobalMemory& operator=(const GlobalMemory& other);

            virtual ~GlobalMemory();

            virtual uint32_t size() const override;

            template<typename T>
            explicit GlobalMemory(const Runtime& runtime, uint count, AccessMode access = AccessMode::ReadWrite)
            : GlobalMemory(runtime, count*sizeof(T), access)
            { }

            template<typename T>
            explicit GlobalMemory(const Runtime& runtime, void* data, uint count, AccessMode access = AccessMode::ReadWrite)
            : GlobalMemory(runtime, data, count*sizeof(T), access)
            { }

            void retrieve(void* data, uint buffer_offset, uint size_to_retrieve) const;

            template<typename T>
            void retrieve(std::vector<T>& dst, uint buffer_offset, uint size_to_retrieve) const
            {
                if(dst.size()*sizeof(T) < size_to_retrieve)
                    dst.resize(size_to_retrieve/sizeof(T));
                retrieve(static_cast<void*>(dst.data()), buffer_offset, size_to_retrieve);
            }

            void write(void* data, uint offset, uint data_size) const;
            void copy_to(const GlobalMemory& other) const;
            void copy_to(const GlobalMemory& other, uint32_t src_offset, uint32_t dst_offset, uint32_t size) const;

            template<typename T>
            void fill(T val, uint32_t offset, uint32_t size)
            {
                fill(&val, sizeof(T), offset, size);
            }

            void fill(void* pattern, uint32_t pattern_size, uint32_t offset, uint32_t size) const;

        protected:

            GlobalMemory(const Runtime& runtime, cl_mem buffer, uint size);

            virtual void bind_to_kernel(const Kernel& k, int idx) const override;

            Runtime _runtime;
            cl_mem  _buffer;
    };
}
