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

#include <libcommon/libcommon.hpp>
#include <libocl/CLGlobal.hpp>
#include <CL/cl.h>

#define CL_CHECK_RET(exp) \
    { \
        int ___status = (exp); \
        if(___status != CL_SUCCESS) \
        { \
            throw ocl::CLException(___status, __LINE__, __FILE__, __FUNCTION__, "CL API Error %d %s", ___status, ocl::CLException::getErrorString(___status)); \
        } \
    }

namespace ocl
{
    /**
     * @class CLException
     * @brief This Exception class represents an OpenCL API error..
     * It inherits from \ref libcommon Exception and keeps the API status error code.
     */
    class CLException : public Exception
    {
        public:
            /**
             * @brief Constructor function
             * @param msg_fmt    Message format string. The message is built using vsnprintf.
             */
            CLException(cl_int status, const char* msg_fmt, ...);

            /**
             * @brief Constructor function
             * @param line      Line number where the exception was thrown.
             * @param file      File name where the exception was thrown.
             * @param func      Function name where the exception was thrown.
             * @param msg_fmt   Message format string.
             */
            CLException(cl_int status, int line, const char* file, const char* func, const char* msg_fmt, ...);

            /**
             * @brief Destructor function
             */
            ~CLException() noexcept(true) = default;

            /**
             * @brief Copy constructor
             */
            CLException(const CLException&) = default;

            /**
             * @brief Copy assignment operator
             */
            CLException& operator=(const CLException&) = default;

            /**
             * @brief This function returns the OpenCL API status error code captured when the exception was thrown.
             * @return      The OpenCL API status error code captured when the exception was thrown.
             */
            int status() const;

            /**
             * @brief This function returns the string representation of the OpenCL API status error code.
             * @param error     The OpenCL API status error code.
             * @return          The string representation of the OpenCL API status error code.
             */
            static const char* getErrorString(int error);

        protected:
            int _status;
    };
}


