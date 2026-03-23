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
	 * Represents the execution of a kernel.
	 * Designed with the builder pattern.
	 */
	class Task final : public NonCopyable
	{
		public:

			Task(const Runtime& runtime, const Kernel& kernel);

			Task& set_global_work_size(uint x, uint y=0, uint z=0);
			Task& set_global_work_offset(uint x, uint y=0, uint z=0);
			Task& set_local_work_size(uint x, uint y=0, uint z=0);
			Task& set_blocking(bool);
			Runtime& runtime();

			/*
			template <const RemoteArg&... args>
			void execute()
			{
				std::vector<const RemoteArg&> arg_list = { args... };

				for(int i = 0; i < arg_list.size(); ++i)
				{
					arg_list[i].bindToKernel(_kernel, 0);
				}
			}
			*/

			// generates signatures void execute(const RemoteArg& arg0 ...)
			SIGNATURES(10);

			void wait();

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
