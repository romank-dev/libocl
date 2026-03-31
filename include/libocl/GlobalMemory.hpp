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
    /**
     * @class GlobalMemory
     * @brief This class represents a global memory buffer in OpenCL.
     */
    class GlobalMemory : public Argument
    {
        public:

            /**
             * @brief Constructor function. Creates a global memory buffer of the specified size and access mode.
             * @param runtime   The runtime object that manages the OpenCL context and command queue.
             * @param size      The size of the buffer in bytes.
             * @param access    The access mode of the buffer. It can be ReadOnly, WriteOnly, or ReadWrite. The default is ReadWrite.
             */
            GlobalMemory(const Runtime& runtime, uint size, AccessMode access = AccessMode::ReadWrite);

            /**
             * @brief Constructor function. Creates a global memory buffer and initializes it with the provided data.
             * @param runtime   The runtime object that manages the OpenCL context and command queue.
             * @param data      A pointer to the data that will be copied to the buffer.
             * @param size      The size of the data in bytes.
             * @param access    The access mode of the buffer. It can be ReadOnly, WriteOnly, or ReadWrite. The default is ReadWrite.
             */
            GlobalMemory(const Runtime& runtime, void* data, uint size, AccessMode access = AccessMode::ReadWrite);

            /**
             * @brief Copy constructor.
             */
            GlobalMemory(const GlobalMemory& other);

            /**
             * @brief Copy assignment operator.
             */
            GlobalMemory& operator=(const GlobalMemory& other);

            /**
             * @brief Destructor function.
             */
            virtual ~GlobalMemory();

            /**
             * @brief Returns the size of the buffer in bytes.
             * @return The size of the buffer in bytes.
             */
            virtual uint32_t size() const override;

            /**
             * @brief Template constructor function. Creates a global memory buffer for an array of the specified type and count.
             * The size of the buffer is calculated as count * sizeof(T).
             * @tparam T        The type of the elements in the array.
             * @param runtime   The runtime object that manages the OpenCL context and command queue.
             * @param count     The number of elements in the array.
             * @param access    The access mode of the buffer. It can be ReadOnly, WriteOnly, or ReadWrite. The default is ReadWrite.
             */
            template<typename T>
            explicit GlobalMemory(const Runtime& runtime, uint count, AccessMode access = AccessMode::ReadWrite)
            : GlobalMemory(runtime, count*sizeof(T), access)
            { }

            /**
             * @brief Template constructor function. Creates a global memory buffer for an array of the specified type and count, and initializes it with the provided data.
             * The size of the buffer is calculated as count * sizeof(T).
             * @tparam T        The type of the elements in the array.
             * @param runtime   The runtime object that manages the OpenCL context and command queue.
             * @param data      A pointer to the data that will be copied to the buffer.
             * @param count     The number of elements in the array.
             * @param access    The access mode of the buffer. It can be ReadOnly, WriteOnly, or ReadWrite. The default is ReadWrite.
             */
            template<typename T>
            explicit GlobalMemory(const Runtime& runtime, void* data, uint count, AccessMode access = AccessMode::ReadWrite)
            : GlobalMemory(runtime, data, count*sizeof(T), access)
            { }

            /**
             * @brief Retrieves data from the buffer and copies it to the provided pointer.
             * The data is copied from the buffer starting at the specified offset and for the specified size.
             * @param data              A pointer to the memory where the retrieved data will be copied.
             * @param buffer_offset     The offset in the buffer from where the data will be retrieved, in bytes.
             * @param size_to_retrieve  The size of the data to retrieve, in bytes.
             */
            void retrieve(void* data, uint buffer_offset, uint size_to_retrieve) const;

            /**
             * @brief Template function that retrieves data from the buffer and copies it to the provided vector.
             * The data is copied from the buffer starting at the specified offset and for the specified size.
             * If the size of the vector is less than the size to retrieve, the vector is resized to accommodate the retrieved data.
             * @tparam T                The type of the elements in the vector.
             * @param dst               A reference to the vector where the retrieved data will be copied.
             * @param buffer_offset     The offset in the buffer from where the data will be retrieved, in bytes.
             * @param size_to_retrieve  The size of the data to retrieve, in bytes.
             */
            template<typename T>
            void retrieve(std::vector<T>& dst, uint buffer_offset, uint size_to_retrieve) const
            {
                if(dst.size()*sizeof(T) < size_to_retrieve)
                    dst.resize(size_to_retrieve/sizeof(T));
                retrieve(static_cast<void*>(dst.data()), buffer_offset, size_to_retrieve);
            }

            /**
             * @brief Writes data to the buffer from the provided pointer.
             * The data is copied to the buffer starting at the specified offset and for the specified size.
             * @param data          A pointer to the memory where the data to be written is stored.
             * @param buffer_offset The offset in the buffer where the data will be written, in bytes.
             * @param data_size     The size of the data to write, in bytes.
             */
            void write(void* data, uint offset, uint data_size) const;

            /**
             * @brief Copy the contents of this buffer to another buffer.
             * @param other         The destination buffer to which the contents of this buffer will be copied.
             */
            void copy_to(const GlobalMemory& other) const;

            /**
             * @brief Copy a portion of this buffer to a portion of another buffer.
             * The data is copied from this buffer starting at the specified source offset and for the specified size, to the destination buffer starting at the specified destination offset.
             * @param other         The destination buffer to which the contents of this buffer will be copied.
             * @param src_offset    The offset in this buffer from where the data will be copied, in bytes.
             * @param dst_offset    The offset in the destination buffer where the data will be copied, in bytes.
             * @param size          The size of the data to copy, in bytes.
             */
            void copy_to(const GlobalMemory& other, uint32_t src_offset, uint32_t dst_offset, uint32_t size) const;


            /**
             * @brief Template function that fills the buffer with a specified value.
             * The value is copied to the buffer starting at the specified offset and for the specified size.
             * @tparam T                The type of the value to fill.
             * @param val               The value to fill the buffer with.
             * @param buffer_offset     The offset in the buffer where the filling will start, in bytes.
             * @param size              The size of the area to fill, in bytes.
             */
            template<typename T>
            void fill(T val, uint32_t offset, uint32_t size)
            {
                fill(&val, sizeof(T), offset, size);
            }

            /**
             * @brief Fills the buffer with a specified pattern.
             * The pattern is copied to the buffer starting at the specified offset and for the specified size.
             * @param pattern           A pointer to the memory where the pattern to fill is stored.
             * @param pattern_size      The size of the pattern, in bytes. The pattern will be repeated to fill the specified size.
             * @param buffer_offset     The offset in the buffer where the filling will start, in bytes.
             * @param size              The size of the area to fill, in bytes.
             */
            void fill(void* pattern, uint32_t pattern_size, uint32_t offset, uint32_t size) const;

        protected:

            GlobalMemory(const Runtime& runtime, cl_mem buffer, uint size);

            virtual void bind_to_kernel(const Kernel& k, int idx) const override;

            Runtime _runtime;
            cl_mem  _buffer;
    };
}
