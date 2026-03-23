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

#include <libocl/Device.hpp>

namespace ocl
{
	/*
	 * Represents a "runnable" context of OpenCL. Runtime is bound to a Device
	 * and is required to perform tasks on the Device (managing memory and executing kernels)
	 */
	// In layman's terms, this class wraps an OpenCL command queue
    class Runtime final
    {
        public:
            explicit Runtime(const Device& device);
            ~Runtime();

            Runtime(const Runtime& other);
            Runtime& operator=(const Runtime& other);

            Device& device();
            const Device& device() const;

        private:
            friend class Argument;
            friend class GlobalMemory;
            friend class OpenCVMat;
            friend class TextureBase;
            friend class Texture2D;
            friend class Texture2DArray;
            friend class Task;

            Device              _device;

            cl_command_queue    _command_queue;
    };

}
