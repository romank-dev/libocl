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

#include <libocl/Argument.hpp>

namespace ocl
{
    /**
     * @class Scalar
     * @brief Represents a scalar value argument to be passed to an OpenCL kernel. It is used for passing simple data types (e.g., int, float, etc.) as kernel arguments.
     * The value is stored on the host and passed by value to the kernel when binding the argument.
     */
	template<typename T>
    class ScalarArg : public Argument
    {
        public:
	        /**
             * @brief Constructor. Creates a RemoteValue argument with the specified value.
             * @tparam T The type of the scalar value (e.g., int, float, etc.).
             * @param value The scalar value to be passed to the kernel.
             */
            explicit ScalarArg<T>(const T& value)
            {
                _value = value;
            }

            /**
             * @brief Get the size of the RemoteValue argument in bytes. This is the size of the underlying type T.
             * @return The size of the RemoteValue argument in bytes.
             */
            virtual uint32_t size() const override
            {
                return sizeof(_value);
            }

            /**
             * @brief Get the value of the RemoteValue argument.
             * @return The scalar value stored in the RemoteValue argument.
             */
            T value() const
            {
                return _value;
            }

            /**
             * Copy constructor
             */
            ScalarArg<T>(const ScalarArg<T>& other)
            {
                _value = other._value;
            }

            /**
             * Copy assignment operator
             */
            ScalarArg<T>& operator=(const ScalarArg<T>& other)
            {
                if(this != &other)
                    _value = other._value;
                return *this;
            }

        private:

            virtual void bind_to_kernel(const Kernel& k, int idx) const override
            {
                CL_CHECK_RET(clSetKernelArg(k._kernel, idx, this->size(), &_value));
            }

            T _value;
    };

}
