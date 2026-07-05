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

#include <iostream>
#include <libocl/Kernel.hpp>
#include <libocl/Runtime.hpp>

#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <libocl/Argument.hpp>

#define ARG(z, n, text) BOOST_PP_COMMA_IF(n) text arg ## n
#define ARGLIST(n) BOOST_PP_REPEAT(n, ARG, const ocl::Argument&)
#define SIG(z, n, text) void execute(ARGLIST(n));
#define SIGNATURES(n) BOOST_PP_REPEAT(n, SIG,)

namespace ocl
{
    /**
     * @class Task
     * @brief Represents the execution of a kernel.
     * It is designed with the builder pattern, allowing you to set various parameters for the kernel execution,
     * such as global and local work sizes, and then execute the kernel with the specified arguments.
     */
	class Task final : public NonCopyable
	{
		public:

	        /**
             * @brief Constructor. Creates a Task for the specified Runtime and Kernel. The Task is associated with the Runtime's Device and the specified Kernel.
             * @param runtime The Runtime context in which the kernel will be executed.
             * @param kernel The Kernel to be executed.
             */
			Task(const Runtime& runtime, const Kernel& kernel);

			/**
			 * @brief Set the global work size for the kernel execution. This specifies the total number of work-items that will execute the kernel.
			 * @param x The global work size in the X dimension.
			 * @param y The global work size in the Y dimension (default is 0, which means it will be ignored).
			 * @param z The global work size in the Z dimension (default is 0, which means it will be ignored).
			 */
			Task& set_global_work_size(uint x, uint y=0, uint z=0);

			/**
             * @brief Set the global work offset for the kernel execution. This specifies the offset for the global IDs of the work-items.
             * @param x The global work offset in the X dimension.
             * @param y The global work offset in the Y dimension (default is 0, which means it will be ignored).
             * @param z The global work offset in the Z dimension (default is 0, which means it will be ignored).
             */
			Task& set_global_work_offset(uint x, uint y=0, uint z=0);

			/**
             * @brief Set the local work size for the kernel execution. This specifies the number of work-items in each work-group.
             * @param x The local work size in the X dimension.
             * @param y The local work size in the Y dimension (default is 0, which means it will be ignored).
             * @param z The local work size in the Z dimension (default is 0, which means it will be ignored).
             */
			Task& set_local_work_size(uint x, uint y=0, uint z=0);

			/**
             * @brief Set whether the kernel execution should be blocking or non-blocking. If set to true, the execute() method will block until the kernel execution is complete.
             * If set to false, the execute() method will return immediately and the kernel execution will happen asynchronously.
             * @param blocking A boolean value indicating whether the kernel execution should be blocking (true) or non-blocking (false).
             * @return A reference to the Task object, allowing for method chaining.
             */
			Task& set_blocking(bool blocking);

			/**
			 * @brief returns a reference to the Runtime associated with this Task.
			 * @return A reference to the Runtime object associated with this Task.
			 */
			Runtime& runtime();

			/*
			template <const Argument&... args>
			void execute()
			{
				std::vector<const Argument&> arg_list = { args... };

				for(int i = 0; i < arg_list.size(); ++i)
				{
					arg_list[i].bindToKernel(_kernel, 0);
				}
			}
			*/

			/**
			 *  generates signatures void execute(const Argument& arg0 ...)
			 *  @param arg0, arg1, ..., arg9 are the arguments to be passed to the kernel. The number of arguments can be up to 10.
			 *  They must be of type const Argument&.
			 *  The execute() method will bind these arguments to the kernel and then run the kernel with the specified work sizes and offsets.
			 */
			SIGNATURES(10);


		private:

			void run_inner();

		private:

			Kernel 		_kernel;
			Runtime 	_runtime;
			size_t 		_global_offsets[3];
			size_t 		_global_work_size[3];
			size_t 		_local_work_size[3];
			bool 		_blocking;

	};
}
