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

#include <libocl/CLGlobal.hpp>
#include <CL/cl.h>
#include <functional>
#include <vector>

namespace ocl
{
    /**
     * @class Device
     * @brief This class represents an OpenCL device. It encapsulates the OpenCL platform and device.
     * It is used to create OpenCL context and resources.
     */
    class Device
    {
        public:

            /**
             * @struct DeviceInfo
             * @brief This struct represents the information about an OpenCL device. It is used to filter and select devices.
             */
            struct DeviceInfo
            {
                std::string platform_name;
                std::string platform_version;
                std::string device_name;
                bool is_gpu;
                bool image_support;
                bool is_64_bit;

                private:
                    friend class Device;
                    DeviceInfo(const std::string& platform_name,
                               const std::string& platform_version,
                               const std::string& device_name,
                               bool is_gpu, bool image_support, bool is_64_bit,
                               cl_platform_id platform, cl_device_id device);

                private:
                    cl_platform_id  _platform;
                    cl_device_id    _device;
            };

        public:

            /**
             * @brief Constructor
             * @param info  The information about the device. The device is created based on the platform and device IDs in the info struct.
             */
            explicit Device(const DeviceInfo& info);

            /**
             * @brief Destructor
             */
            ~Device();

            /**
             * @brief Copy constructor
             */
            Device(const Device& other);

            /**
             * @brief Copy assignment operator
             */
            Device& operator=(const Device& other);

            /**
             * @brief This function returns the information about all the available devices.
             * @return vector of DeviceInfo structs containing the information about all the available devices.
             */
            static std::vector<DeviceInfo> get_all_devices_info();

            /**
             * @brief This function returns the information about the first device that satisfies the given filter.
             * @param filter    The filter function to select the device.
             *                  It takes a DeviceInfo struct and returns a boolean value indicating whether the device satisfies the filter criteria.
             * @note If no device satisfies the filter criteria, an exception is thrown.
             */
            static Device get_device_using_filter(const std::function<bool(const DeviceInfo&)>& filter);

            /**
             * @brief This function returns the first GPU device available on the system. If no GPU device is found, an exception is thrown.
             * @return The first GPU device available on the system.
             */
            static Device get_any_gpu_device();

        private:
            Device(cl_platform_id platform, cl_device_id device);

            static Device get_gpu_using_device_filter(const std::function<bool(const std::string&)>&);

            cl_platform_id  _platform;
            cl_device_id    _device;
            cl_context      _context;

        private:

            friend class Runtime;
            friend class Kernel;
            friend class Argument;
            friend class GlobalMemory;
            friend class OpenCVMat;
            friend class TextureBase;
            friend class Texture2D;
            friend class Texture2DArray;
    };
}
