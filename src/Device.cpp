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

#include <libocl/Device.hpp>
#include <libocl/CLException.hpp>
#include "logging.hpp"

using namespace ocl;
using namespace std;

Device::Device(cl_platform_id platform, cl_device_id device)
    : _platform(platform), _device(device)
{
    cl_context_properties props[4] = {};
    props[0] = CL_CONTEXT_PLATFORM;
    props[1] = (cl_context_properties)platform;
    cl_int ret_code;
    cl_context context =
            clCreateContext(
                props,
                1,
                &device,
                [](const char *errinfo, const void *private_info, size_t cb, void *user_data)
                {
                    TRACE_ERR("OCL ERROR CALLBACK: %s",errinfo);
                },
                nullptr,
                &ret_code);
    CL_CHECK_RET(ret_code);
    _context = context;
}

Device::Device(const Device& other)
    : _device(other._device), _context(other._context), _platform(other._platform)
{
    CL_CHECK_RET(clRetainContext(_context));
}


Device& Device::operator=(const Device& other)
{
    if(this == &other)
        return *this;

    _device = other._device;
    _platform = other._platform;

    CL_CHECK_RET(clRetainContext(other._context));
    CL_CHECK_RET(clReleaseContext(_context));

    _context = other._context;

    return *this;
}

Device::~Device()
{
    SAFE_DESTRUCTOR
    (
        CL_CHECK_RET(clReleaseContext(_context));
    );
}

Device Device::get_device_using_filter(const std::function<bool(const DeviceInfo&)>& filter)
{
    auto devices = get_all_devices_info();
    for(const auto& device_info : devices)
    {
        if(filter(device_info))
        {
            return Device(device_info._platform, device_info._device);
        }
    }
    throw Exception("No appropriate OpenCL device found");
}

Device Device::get_any_gpu_device()
{
    return get_device_using_filter([](const DeviceInfo& info)
    {
        return info.is_gpu;
    });
}

vector<Device::DeviceInfo> Device::get_all_devices_info()
{
    vector<DeviceInfo> res;

    cl_uint num_platforms;
    CL_CHECK_RET(clGetPlatformIDs(0, nullptr, &num_platforms));

    vector<cl_platform_id> platforms(num_platforms);
    CL_CHECK_RET(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

    for(int i = 0; i < num_platforms; i++)
    {
        size_t name_len;
        CL_CHECK_RET(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &name_len));

        string platform_name(name_len, 0);
        CL_CHECK_RET(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, name_len, (void*)platform_name.data(), nullptr));

        size_t version_len;
        CL_CHECK_RET(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, 0, nullptr, &version_len));

        string platform_version(version_len, 0);
        CL_CHECK_RET(clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, version_len, (void*)platform_version.data(), nullptr));

        cl_uint num_devices;
        CL_CHECK_RET(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices));

        vector<cl_device_id> device_ids(num_devices);
        CL_CHECK_RET(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, device_ids.size(), device_ids.data(), nullptr));

        for(int j = 0; j < num_devices; j++)
        {
            cl_device_type device_type;
            CL_CHECK_RET(clGetDeviceInfo(device_ids[j], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr));

            size_t device_name_len;
            CL_CHECK_RET(clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, 0, nullptr, &device_name_len));

            Buffer device_name(device_name_len, 0);
            CL_CHECK_RET(clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, device_name_len, (void*)device_name.data(), nullptr));

            cl_bool image_support;
            CL_CHECK_RET(clGetDeviceInfo(device_ids[j], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &image_support, nullptr));

            cl_uint address_bits;
            CL_CHECK_RET(clGetDeviceInfo(device_ids[j], CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), &address_bits, nullptr));

            DeviceInfo info(
                    platform_name,
                    platform_version,
                    (const char*)device_name.data(),
                    (device_type == CL_DEVICE_TYPE_GPU),
                    (image_support == CL_TRUE),
                    (address_bits == 64),
                    platforms[i], device_ids[j]);

            res.push_back(info);
        }
    }

    return res;
}

Device::DeviceInfo::DeviceInfo(const std::string& platform_name,
                       const std::string& platform_version,
                       const std::string& device_name,
                       bool is_gpu,
                       bool image_support,
                       bool is_64_bit,
                       cl_platform_id platform,
                       cl_device_id device)
    : platform_name(platform_name),
      platform_version(platform_version),
      device_name(device_name),
      is_gpu(is_gpu),
      image_support(image_support),
      is_64_bit(is_64_bit),
      _platform(platform),
      _device(device)
{}

