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
	/**
     * @class Kernel
     * @brief Represents a compiled form of an OpenCL Kernel.
     * It is bound to a Device for which the kernel is compiled.
     * The compilation is done by the driver and does not address the Device yet.
     */
    class Kernel
    {
        friend class Argument;
        friend class LocalMemory;
    	template<typename T> friend class ScalarArg;
        friend class GlobalMemory;
        friend class TextureBase;
        friend class Texture2D;
        friend class Texture2DArray;
        friend class Task;

        private:
            /**
             * @class Program
             * @brief Represents a compiled OpenCL program. It is used internally by the Kernel class to manage the compilation of the kernel source code.
             */
            class Program
            {
                public:
                    /**
                     * @brief Constructor. Compiles the OpenCL program from the given source file and function name, using the specified preprocessor definitions.
                     * @param device The Device for which the program is compiled.
                     * @param file_name The name of the source file containing the kernel code.
                     * @param func_name The name of the kernel main function.
                     * @param preprocessor A map of preprocessor definitions to be used during compilation. (-D flag in OpenCL compiler)
                     */
                    Program(
                            const Device& device,
                            const std::string& file_name,
                            const std::string& func_name,
                            const std::map<std::string, std::string>& preprocessor);
                    ~Program();

                    /**
                     * Copy constructor
                     */
                    Program(const Program& other);

                    /**
                     * Copy assignment operator
                     */
                    Program& operator=(const Program& other);

                    /**
                     * @brief Get the file name of the source code used to compile the program.
                     * @return The file name of the source code.
                     */
                    const std::string& file_name() const;

                    /**
                     * @brief Get the name of the kernel main function.
                     * @return The name of the kernel main function.
                     */
                    const std::string& func_name() const;

                    /**
                     * @brief Get the preprocessor definitions used during compilation.
                     * @return A string representing the preprocessor definitions.
                     */
                    const std::string& preprocessor() const;

                    /**
                     * @brief Assemble the preprocessor definitions into a string that can be passed to the OpenCL compiler.
                     * @return A string representing the preprocessor definitions in the format expected by the OpenCL compiler (e.g., "-DKEY=VALUE -DKEY2=VALUE2").
                     */
                    static std::string assemble_preprocessor_string(const std::map<std::string, std::string>& preprocessor);

                    /**
                     * @brief Implicit conversion to get the compiled OpenCL program object.
                     * @return The OpenCL program object (cl_program).
                     */
                    cl_program operator()();

                private:
                    std::string _file_name;
                    std::string _func_name;
                    std::string _preprocessor;
                    cl_program  _program;
            };

        public:

            /**
             * @brief Constructor. Compiles the kernel for the specified device, source file, function name, and preprocessor definitions.
             * @param device The Device for which the kernel is compiled.
             * @param file_name The name of the source file containing the kernel code.
             * @param func_name The name of the kernel main function.
             * @param preprocessor A map of preprocessor definitions to be used during compilation. (-D flag in OpenCL compiler)
             */
            Kernel(
                    const Device& device,
                    const std::string& file_name,
                    const std::string& func_name,
                    const std::map<std::string, std::string>& preprocessor = {}
                    );

            /**
             * Destructor
             */
            ~Kernel();

            /**
             * @brief Get the number of arguments that the kernel function takes.
             * @return The number of arguments of the kernel function.
             */
            uint num_args() const;

            /**
             * @brief Recompile the kernel if the preprocessor definitions have changed. This is useful when you want to change the
             * behavior of the kernel by modifying the preprocessor definitions without changing the source code.
             * @param preprocessor A map of preprocessor definitions to be used during compilation. (-D flag in OpenCL compiler)
             */
            void recompile_if_needed(const std::map<std::string, std::string>& preprocessor);

            /**
             * @brief Print information about the kernel, such as the number of arguments, work group size, and private memory usage.
             * This can be useful for debugging and performance tuning.
             */
            void print_info() const;

            /**
             * Copy constructor
             */
            Kernel(const Kernel& other);

            /**
             * Copy assignment operator
             */
            Kernel& operator=(const Kernel& other);

            /**
             * @brief Get the compiled program associated with this kernel.
             * @return The Program object associated with this kernel.
             */
            const Program& program() const;

        private:
            Device      _device;
            Program     _program;
            cl_kernel   _kernel;
            cl_uint		_num_args;
    };
}


