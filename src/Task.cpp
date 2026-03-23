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


#include <libocl/Task.hpp>
#include "logging.hpp"

using namespace ocl;

#define BIND(z, n, test) arg##n.bind_to_kernel(_kernel, n);

#define IMP(z, n, text) \
void Task::execute(ARGLIST(n))\
{\
	CHECK_THROW(n == _kernel.num_args(), "Incompatible argument count! (Kernel: %d, You gave: %d)", _kernel.num_args(), n);\
	BOOST_PP_REPEAT(n, BIND,) \
	run_inner(); \
}

#define IMPLEMENTATIONS(n) BOOST_PP_REPEAT(n, IMP,)

Task::Task(const Runtime& runtime, const Kernel& kernel)
	: _runtime(runtime), _kernel(kernel), _global_offsets{}, _global_work_size{}, _local_work_size{}, _blocking(true)
{
}

IMPLEMENTATIONS(10)

void Task::run_inner()
{
	CHECK_THROW(_global_work_size[0] > 0, "Work size was not set!");
	int dims = 1;
	dims += _global_work_size[1] > 0;
	dims += _global_work_size[2] > 0;

	try
	{
	    CL_CHECK_RET(clEnqueueNDRangeKernel(_runtime._command_queue, _kernel._kernel, dims, _global_offsets,
            _global_work_size, _local_work_size[0] == 0 ? nullptr : _local_work_size, 0, nullptr, nullptr));

	    if(_blocking)
            wait();
	}
	catch(const Exception& e)
	{
	    throw Exception("Kenrel %s:%s failed: %s", _kernel.program().file_name().c_str(), _kernel.program().func_name().c_str(), e.what());
	}
}

Task& Task::set_global_work_offset(uint x, uint y, uint z)
{
	_global_offsets[0] = x;
	_global_offsets[1] = y;
	_global_offsets[2] = z;
	return *this;
}

Task& Task::set_global_work_size(uint x, uint y, uint z)
{
	_global_work_size[0] = x;
	_global_work_size[1] = y;
	_global_work_size[2] = z;
	return *this;
}

Task& Task::set_local_work_size(uint x, uint y, uint z)
{
	_local_work_size[0] = x;
	_local_work_size[1] = y;
	_local_work_size[2] = z;
	return *this;
}

Task& Task::set_blocking(bool flag)
{
	_blocking = flag;
	return *this;
}

Runtime& Task::runtime()
{
    return _runtime;
}

void Task::wait()
{
	CL_CHECK_RET(clFlush(_runtime._command_queue));
	CL_CHECK_RET(clFinish(_runtime._command_queue));
}
