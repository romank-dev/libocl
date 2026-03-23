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
#include <libocl/CLException.hpp>
#include <libocl/Device.hpp>
#include <map>

namespace ocl
{
	/* Represents a compiled form of an OpenCL Kernel.
	 * It is bound to a Device for which the kernel is compiled.
	 * The compilation is done by the driver and does not address the Device yet.
	 */
    class Kernel
    {
        friend class Argument;
        friend class LocalMemory;
    	template<typename T> friend class RemoteValue;
        friend class GlobalMemory;
        friend class TextureBase;
        friend class Texture2D;
        friend class Texture2DArray;
        friend class Task;

        private:
            class Program
            {
                public:
                    Program(
                            const Device& device,
                            const std::string& file_name,
                            const std::string& func_name,
                            const std::map<std::string, std::string>& preprocessor);
                    ~Program();

                    Program(const Program& other);
                    Program& operator=(const Program& other);

                    const std::string& file_name() const;
                    const std::string& func_name() const;
                    const std::string& preprocessor() const;

                    static std::string assemble_preprocessor_string(const std::map<std::string, std::string>& preprocessor);
                    cl_program operator()();

                private:
                    std::string _file_name;
                    std::string _func_name;
                    std::string _preprocessor;
                    cl_program  _program;
            };

        public:
            Kernel(
                    const Device& device,
                    const std::string& file_name,
                    const std::string& func_name,
                    const std::map<std::string, std::string>& preprocessor = {}
                    );

            ~Kernel();

            uint num_args() const;

            void recompile_if_needed(const std::map<std::string, std::string>& preprocessor);

            void print_info() const;

            Kernel(const Kernel& other);
            Kernel& operator=(const Kernel& other);

            const Program& program() const;

        private:
            Device      _device;
            Program     _program;
            cl_kernel   _kernel;
            cl_uint		_num_args;
    };
}


