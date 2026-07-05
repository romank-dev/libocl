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


#include <map>
#include <functional>
#include <thread>
#include <gtest/gtest.h>

#include <libocl/libocl.hpp>

using namespace ocl;
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}


TEST(LibOcl, TestSanity)
{
    Utils::FileSystem::set_local_cwd();

    Device device = Device::get_any_gpu_device();

    Runtime runtime(device);

    Kernel kernel(device, "bayer.cl", "bayer_to_rgb");

    Texture2D tex(runtime, 200, 300, ImageFormat::RGBA, AccessType::Int8N);

    ScalarArg<int> val(5);

    GlobalMemory buf(runtime, 1024000, AccessMode::ReadWrite);

    OpenCVMat rmat(runtime, cv::Size(1000,250), CV_8UC3, AccessMode::ReadOnly);
}

TEST(LibOcl, TestBayerToRGB)
{
    Device device = Device::get_any_gpu_device();
    Runtime runtime(device);

    Kernel kernel(device, "bayer.cl", "bayer_to_rgb");

    cv::Mat img = cv::imread("test.png", cv::IMREAD_GRAYSCALE);

    Texture2D rggb(runtime, img, AccessType::Uint8, AccessMode::ReadOnly);
    Texture2D rgba(runtime, img.cols, img.rows, ImageFormat::RGBA, AccessType::Uint8);

    Task t(runtime, kernel);
    t.set_global_work_size(img.cols, img.rows);
    t.execute(rggb, rgba);


    cv::Mat out = rgba.retrieve_mat();
    cv::Mat test = cv::imread("bayer.png", cv::IMREAD_COLOR);
    cvtColor(test, test, COLOR_BGR2BGRA);

    ASSERT_TRUE(cv::countNonZero(out != test) == 0);

    //imshow("res", out);
    //imshow("res2", test);
    waitKey(0);
}


TEST(LibOcl, TestRGBAtoRGB_OpenCvVsOpenCL)
{
    Utils::FileSystem::set_local_cwd();

    Device device = Device::get_any_gpu_device();
    Runtime runtime(device);
    Kernel kernel(device, "rgba2rgb.cl", "rgba_to_rgb");

    cv::Mat img = cv::imread("test2.png", cv::IMREAD_COLOR);
    cv::Mat img_rgba;

    cv::cvtColor(img, img_rgba, cv::COLOR_BGR2BGRA);

    Texture2D rgba(runtime, img_rgba, AccessType::Uint8, AccessMode::ReadOnly);
    OpenCVMat rgb(runtime, cv::Size(img_rgba.cols, img_rgba.rows), CV_8UC3);

    Task task(runtime, kernel);
    task.set_global_work_size(img.cols, img.rows);
    task.set_blocking(false);

    int number_of_runs = 1000;
    Stopwatch t;
    cv::Mat img_rgb;
    cv::setNumThreads(1); // comment this to have a fair fight
    for (int i = 0; i < number_of_runs; i++)
    {
        cv::cvtColor(img_rgba, img_rgb, cv::COLOR_BGRA2BGR);
    }

    printf("OpenCV took %lu us\n", t.get_us()/number_of_runs);

    cv::Mat out;
    cv::Mat out2(img_rgba.rows, img_rgba.cols, CV_8UC3, cv::Scalar(0));
    t.reset();
    for (int i = 0; i < number_of_runs; i++)
    {
        task.execute(rgba, rgb);
    }
    runtime.wait_for_task_completion();
    rgb.retrieve(out2);
    printf("OpenCL took %lu us\n", t.get_us()/number_of_runs);

    ASSERT_TRUE(cv::countNonZero(out2 != img_rgb) == 0);
}

