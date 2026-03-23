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
	template<typename T>
    class RemoteValue : public Argument
    {
        public:
            explicit RemoteValue<T>(const T& value)
            {
                _value = value;
            }

            virtual ~RemoteValue()
            {
            }

            virtual uint32_t size() const override
            {
                return sizeof(_value);
            }

            RemoteValue<T>(const RemoteValue<T>& other)
            {
                _value = other._value;
            }

            T value() const
            {
                return _value;
            }

            RemoteValue<T>& operator=(const RemoteValue<T>& other)
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
