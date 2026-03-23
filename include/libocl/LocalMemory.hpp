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
	class LocalMemory : public Argument
    {
        public:
            explicit LocalMemory(uint size);
            virtual ~LocalMemory() = default;

            virtual uint32_t size() const override;

            LocalMemory(const LocalMemory& other);
            LocalMemory& operator=(const LocalMemory& other);

        private:
            virtual void bind_to_kernel(const Kernel& k, int idx) const override;

        private:
            uint32_t    _size;
    };

}
