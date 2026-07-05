# libocl
OpenCL C++ wrapper library
## Building
* **libocl** is built using the [workspace](https://github.com/romank-dev/workspace) build system. 
** For building with a different system, see [this link](https://github.com/romank-dev/workspace?tab=readme-ov-file#deep-dive--building-projects-without-workspace).
* The manifest [URL](https://github.com/romank-dev/libocl/blob/main/manifest.txt) may be used by the Workspace **init.sh** script to pull this project and it's dependencies.

## Overview
**libocl** contains classes that wrap OpenCL's APIs and tools to perform computations on GPUs (and other accelerators).
It allows you to quickly and easily get GPU compuation going thanks to organized and safe C++ wrappers over the OpenCL's resources, taking care of resource management, error checking and handling, leaving only writing the kernel itself as-is to the user.
The library is intended to support OpenCL 1.1, 1.2 and 2.0, where most of the mid to low end hardware is right now.
All of the classes are written with RAII and exception safety built-in, and implement copying by using OpenCL's reference counting APIs. <br/>
The library by no means covers the entire list of resources and tools that OpenCL provides. The current list of classes is:
* Device - Device context. Wrapper for OpenCL's cl_context (as well as cl_device and cl_platform).
* Runtime - Command queue for a device. Wrapper for cl_command_queue.
* Kernel - Compiled, ready to run program for a device. Wraps cl_program and cl_kernel.
* Task - Instance of kernel execution. Gathers arguments, work sizes, and wraps clEnqueueNDRangeKernel, the most important function.
* GlobalMemory - General purpose GPU "global" memory. Wraps cl_mem.
* LocalMemory - Fast memory to be allocatead per workgroup. Synonymous CUDA's shared memory.
* ScalarArg - Wraps a generic scalar argument to a kernel.
* OpenCVMat - Conveniently manages an OpenCV cv::Mat in device memory. Allows safe transition between the host. Uses global memory underneath.
* Texture2D - Manages a texture (image) in device memory. It uses the image memory instead of global memory, optimized for image processing algorithms.
* Texture2DArray - Manages an array of 2D images in device memory. Internally requests an image object of type CL_MEM_OBJECT_IMAGE2D_ARRAY.
  
More to come...


## Demo
Here is a demonstration of how fast this library lets you utilize your GPU. <br/>
We will be converting an RGBA image to and RGB image format in 10 steps (aka lines of code):
```cpp
    Device device = Device::get_any_gpu_device(); // 1. Get any GPU context for demo purposes
    Runtime runtime(device); // 2. Create command queue for the GPU (you can have several)
    Kernel kernel(device, "rgba2rgb.cl", "rgba_to_rgb"); // 3. Load the program rgba2rgb.cl and compile the kernel function rgba_to_rgb
    cv::Mat img_rgba = cv::imread("rgba.png", cv::IMREAD_UNCHANGED); // 4. Read source image from disk
    Texture2D rgba(runtime, img_rgba, AccessType::Uint8, AccessMode::ReadOnly); // 5. Upload source image to GPU memory
    OpenCVMat rgb(runtime, cv::Size(img_rgba.cols, img_rgba.rows), CV_8UC3); // 6. Allocate target image GPU memory
    Task task(runtime, kernel); // 7. Create the GPU task 
    task.set_global_work_size(img.cols, img.rows); // 8. Set work sizes
    task.execute(rgba, rgb); // 9. Execute the task
    cv::Mat img_rgb = rgb.retrieve(); // 10. Retrieve result from GPU memory
```
And here is the simple OpenCL kernel used for this example:
```opencl
    __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

    __kernel void rgba_to_rgb(__read_only image2d_t rgba, __global uchar* rgb)
    {
        int2 pos = (int2)(get_global_id(0), get_global_id(1));
        int w = get_image_width(rgba);
        
        int4 p = read_imagei(rgba, sampler, pos);
        rgb[(pos.x + pos.y * w)*3 + 0] = p.x;
        rgb[(pos.x + pos.y * w)*3 + 1] = p.y;
        rgb[(pos.x + pos.y * w)*3 + 2] = p.z;
    }
```
