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

#include <libocl/Kernel.hpp>
#include <fstream>
#include <string>
#include <sstream>
#include "logging.hpp"

using namespace ocl;
using namespace std;

Kernel::Program::Program(const Device& device, const string& file_name,const string& func_name, const map<string, string>& preprocessor)
    :  _file_name(file_name), _func_name(func_name)
{
    fstream f(file_name.c_str(), (fstream::in | fstream::binary));
    CHECK_THROW(f.is_open(), "Kernel::Program::Program(): Opening file %s failed\n", file_name.c_str());

    size_t fileSize;
    f.seekg(0, fstream::end);
    size_t size = fileSize = (size_t)f.tellg();
    f.seekg(0, fstream::beg);
    vector<char> source(size+1);
    f.read(source.data(), fileSize);
    f.close();
    source[size] = '\0';

    size_t sourceSize[] = {source.size()};
    string strobj(source.data());
    const char* str = strobj.c_str();

    cl_int error_code;
    _program = clCreateProgramWithSource(device._context, 1, &str, sourceSize, &error_code);
    CL_CHECK_RET(error_code);

    cl_device_id dev = device._device;

    _preprocessor = assemble_preprocessor_string(preprocessor);

    TRACE_DBG("Compiling kernel %s/%s",file_name.c_str(),func_name.c_str());
    int compile_status = clBuildProgram(_program, 1, &dev, _preprocessor.c_str(), NULL,NULL);

    ExceptionGuard cleaner([&]()
	{
    	this->~Program();
	});

    if (compile_status == CL_BUILD_PROGRAM_FAILURE)
    {
        // Determine the size of the log
        size_t log_size;
        CL_CHECK_RET(clGetProgramBuildInfo(_program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size));
        vector<char> log(log_size);
        CL_CHECK_RET(clGetProgramBuildInfo(_program, dev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL));
        throw Exception("Kernel::Program::Program(): Compilation error (%s/%s):\n%s\n", file_name.c_str(), func_name.c_str(), log.data());
    }

    CL_CHECK_RET(compile_status); // will throw!
}

const string& Kernel::Program::preprocessor() const
{
    return _preprocessor;
}

string Kernel::Program::assemble_preprocessor_string(const map<string, string>& preprocessor)
{
    stringstream ss;
    for(auto define : preprocessor)
    {
        ss << " -D" << define.first << "=" << define.second;
    }
    return ss.str();
}

void Kernel::recompile_if_needed(const map<string, string>& preprocessor)
{
    string pre = Program::assemble_preprocessor_string(preprocessor);
    if(pre == _program.preprocessor())
        return;
    TRACE_DBG("Recompiling kernel %s/%s",_program.file_name().c_str(),_program.func_name().c_str());

    this->operator =(Kernel(_device, _program.file_name(), _program.func_name(), preprocessor));
}

cl_program Kernel::Program::operator ()()
{
    return _program;
}

const string& Kernel::Program::file_name() const
{
    return _file_name;
}

const string& Kernel::Program::func_name() const
{
    return _func_name;
}

Kernel::Program::Program(const Kernel::Program& other)
{
    CL_CHECK_RET(clRetainProgram(other._program));
    _program = other._program;
    _file_name = other._file_name;
    _func_name = other._func_name;
    _preprocessor = other._preprocessor;
}

const Kernel::Program& Kernel::program() const
{
    return _program;
}

void Kernel::print_info() const
{
    size_t wg_size;
    cl_ulong private_mem;
    cl_ulong local_mem;
    size_t wg_size_multiple;
    size_t wg[3];
    size_t size_ret;
    CL_CHECK_RET(clGetKernelWorkGroupInfo(_kernel, NULL, CL_KERNEL_WORK_GROUP_SIZE, sizeof(wg_size), &wg_size, &size_ret));
    CL_CHECK_RET(clGetKernelWorkGroupInfo(_kernel, NULL, CL_KERNEL_COMPILE_WORK_GROUP_SIZE, sizeof(wg), wg, &size_ret));
    CL_CHECK_RET(clGetKernelWorkGroupInfo(_kernel, NULL, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem, &size_ret));
    CL_CHECK_RET(clGetKernelWorkGroupInfo(_kernel, NULL, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(wg_size_multiple), &wg_size_multiple, &size_ret));
    CL_CHECK_RET(clGetKernelWorkGroupInfo(_kernel, NULL, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(private_mem), &private_mem, &size_ret));

    printf("Workgroup size: %lu (%lux%lux%lu)\nWorkgroup size multiple: %lu\nLocal memory: %lu\nPrivate memory: %lu\n", wg_size, wg[0],wg[1],wg[2], wg_size_multiple, local_mem, private_mem);
}

Kernel::Program& Kernel::Program::operator=(const Kernel::Program& other)
{
    if(this != &other)
    {
        CL_CHECK_RET(clReleaseProgram(_program));
        CL_CHECK_RET(clRetainProgram(other._program));
        _program = other._program;
        _file_name = other._file_name;
        _func_name = other._func_name;
        _preprocessor = other._preprocessor;
    }
    return *this;
}

Kernel::Program::~Program()
{
    SAFE_DESTRUCTOR
    (
        CL_CHECK_RET(clReleaseProgram(_program));
    )
}

Kernel::Kernel(
        const Device& device,
        const string& file_name,
        const string& func_name,
        const map<string, string>& preprocessor
        )
    : _device(device), _program(device, file_name, func_name, preprocessor)
{
    cl_int error_code;
    cl_kernel kernel = clCreateKernel(_program(), func_name.c_str(), &error_code);
    CL_CHECK_RET(error_code);

    _kernel = kernel;

    CL_CHECK_RET(clGetKernelInfo(_kernel, CL_KERNEL_NUM_ARGS, sizeof(_num_args), &_num_args, nullptr));
}

Kernel::Kernel(const Kernel& other)
    : _device(other._device), _program(other._program)
{
    CL_CHECK_RET(clRetainKernel(other._kernel));
    _kernel = other._kernel;
    _num_args = other._num_args;
}

Kernel& Kernel::operator=(const Kernel& other)
{
    if(this != &other)
    {
        CL_CHECK_RET(clReleaseKernel(_kernel));
        CL_CHECK_RET(clRetainKernel(other._kernel));
        _kernel = other._kernel;
        _program = other._program;
    }
    return *this;
}

uint Kernel::num_args() const
{
	return _num_args;
}

Kernel::~Kernel()
{
    SAFE_DESTRUCTOR
    (
        CL_CHECK_RET(clReleaseKernel(_kernel));
    )
}
