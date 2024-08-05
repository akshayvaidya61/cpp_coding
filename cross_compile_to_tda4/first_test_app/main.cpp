#include "hello_world_tda4.h"
#include <vector>
#include <string>
#include <iostream>
#include <thread>
#include <csignal>

#include "opencv2/imgcodecs.hpp"

extern "C"
{
#include "TI/tivx.h"
#include "VX/vxu.h"
#include "VX/vx.h"
#include "processor_sdk/vision_apps/utils/app_init/include/app_init.h"
#include "processor_sdk/tiovx/utils/include/tivx_utils_file_rd_wr.h"
#include "TI/tivx_log_rt.h"
}

bool isExitRequested = false;
void signal_handler(int signal)
{
    isExitRequested = true;
}

int main()
{

    std::signal(SIGINT, signal_handler);

    vx_context context;
    vx_graph graph;
    vx_image input;
    vx_image output;
    vx_node scale_node;
    std::string filename;

    appInit();
    tivxInit();

    if (context = vxCreateContext(); vxGetStatus(reinterpret_cast<vx_reference>(context)) != VX_SUCCESS)
    {
        std::cout << "Failed to create context" << std::endl;
        std::runtime_error("Failed to create context");
    }

    if (graph = vxCreateGraph(context); vxGetStatus(reinterpret_cast<vx_reference>(graph)) != VX_SUCCESS)
    {
        std::cout << "Failed to create graph" << std::endl;
        std::runtime_error("Failed to create graph");
    }

    if (input = vxCreateImage(context, 1280, 960, VX_DF_IMAGE_U8); vxGetStatus(reinterpret_cast<vx_reference>(input)) != VX_SUCCESS)
    {
        std::cout << "Failed to create input image" << std::endl;
        std::runtime_error("Failed to create input image");
    }

    if (auto status = tivxLogRtTraceEnable(graph); status != VX_SUCCESS)
    {
        std::cout << "Failed to enable trace" << std::endl;
    }

    filename = "/tmp/frame.png";
    const vx_rectangle_t rect = {0, 0, 1280, 960};
    vx_imagepatch_addressing_t addr;
    vx_map_id map_id;
    void *ptr;

    if (vxMapImagePatch(input, &rect, 0, &map_id, &addr, &ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0) != VX_SUCCESS)
    {
        std::cout << "Failed to map input image" << std::endl;
        std::runtime_error("Failed to map input image");
    }
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (!image.data)
    {
        std::cout << "Failed to read image" << std::endl;
        std::runtime_error("Failed to read image");
    }

    memcpy(ptr, image.data, image.total() * image.elemSize());
    vxUnmapImagePatch(input, map_id);

    if (output = vxCreateImage(context, 640, 480, VX_DF_IMAGE_U8); vxGetStatus(reinterpret_cast<vx_reference>(output)) != VX_SUCCESS)
    {
        std::cout << "Failed to create output image" << std::endl;
        std::runtime_error("Failed to create output image");
    }

    vx_imagepatch_addressing_t addr2;
    vx_map_id map_id2;
    void *ptr2;

    if (vxMapImagePatch(output, &rect, 0, &map_id2, &addr2, &ptr2, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0) != VX_SUCCESS)
    {
        std::cout << "Failed to map output image" << std::endl;
        std::runtime_error("Failed to map output image");
    }

    cv::Mat image2(480, 640, CV_8UC1, ptr2, addr2.stride_y);
    image2.setTo(0);

    memcpy(image2.data, ptr2, image2.total() * image2.elemSize());

    vxUnmapImagePatch(output, map_id);

    if (scale_node = vxScaleImageNode(graph, input, output, VX_INTERPOLATION_BILINEAR); vxGetStatus(reinterpret_cast<vx_reference>(scale_node)) != VX_SUCCESS)
    {
        std::cout << "Failed to create scale node" << std::endl;
        std::runtime_error("Failed to create scale node");
    }

    if (auto result = vxSetNodeTarget(scale_node, VX_TARGET_STRING, TIVX_TARGET_VPAC_MSC1); result != VX_SUCCESS)
    {
        std::cout << "Failed to set node targets" << std::endl;
        std::runtime_error("Failed to set node targets");
    }

    if (auto result = vxVerifyGraph(graph); result != VX_SUCCESS)
    {
        std::cout << "Failed to verify graph" << std::endl;
        std::runtime_error("Failed to verify graph");
    }

    int index = 0;
    while (!isExitRequested)
    {
        if (auto result = vxProcessGraph(graph); result != VX_SUCCESS)
        {
            std::cout << "Failed to process graph" << std::endl;
            std::runtime_error("Failed to process graph");
        }

        cv::imwrite("/tmp/output.png", image2);
        if (index % 100 == 0)
        {
            std::cout << "Process Frame Index: " << index << std::endl;
            tivxLogRtTraceExportToFile("/tmp/trace.log");
        }
        index++;
    }

    vxReleaseNode(&scale_node);
    vxReleaseImage(&input);
    vxReleaseImage(&output);
    vxReleaseGraph(&graph);
    vxReleaseContext(&context);

    tivxDeInit();
    appDeInit();
}
