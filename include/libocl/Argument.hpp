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


#include <libocl/Kernel.hpp>
#include <libocl/Definitions.hpp>

namespace ocl
{
    /**
     * @class Argument
     * @brief This class represents an argument to be passed to an OpenCL kernel.
     * It is an abstract base class that defines the interface for all types of arguments, such as buffers, images, and scalar values.
     */
    class Argument
    {
        public:

            /**
             * @brief The size of the argument in bytes. For scalar types, this is the size of the type.
             * For buffer and image types, this is the size of the data they represent.
             * @return The size of the argument in bytes.
             */
            virtual uint32_t size() const = 0;

        protected:
            virtual ~Argument() = default;

        private:

            friend class Task;
            virtual void bind_to_kernel(const Kernel& k, int idx) const = 0;


    };
}
