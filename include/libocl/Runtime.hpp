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
    /**
     * @class Runtime
     * @brief Represents a "runnable" context of OpenCL. Runtime is bound to a Device and is required to perform
     * tasks on the Device (managing memory and executing kernels). This class wraps an OpenCL command queue.
     */
    class Runtime final
    {
        public:
            /**
             * @brief Constructor. Creates a Runtime context for the specified Device. This involves creating an OpenCL command queue for the device.
             * @param device The Device for which the Runtime context is created.
             */
            explicit Runtime(const Device& device);

            /**
             * Destructor.
             */
            ~Runtime();

            /**
             * Copy constructor.
             */
            Runtime(const Runtime& other);

            /**
             * Copy assignment operator.
             */
            Runtime& operator=(const Runtime& other);

            /**
             * @brief Get the Device associated with this Runtime context.
             * @return The Device associated with this Runtime context.
             */
            Device& device();

            /**
             * @brief Get the Device associated with this Runtime context (const version).
             * @return The Device associated with this Runtime context.
             */
            const Device& device() const;

            /**
             * @brief Wait for all previously enqueued tasks in the command queue to complete.
             * This is a blocking call that ensures that all operations have finished before proceeding.
             */
            void wait_for_task_completion();

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
