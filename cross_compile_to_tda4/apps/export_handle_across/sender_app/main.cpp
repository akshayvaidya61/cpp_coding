#include <vector>
#include <iostream>
#include <thread>
#include <csignal>

extern "C"
{
#include "TI/tivx.h"
#include "VX/vxu.h"
#include "VX/vx.h"
#include "vision_apps/utils/app_init/include/app_init.h"
#include "tiovx/utils/include/tivx_utils_ipc_ref_xfer.h"
}

bool isExitRequested = false;

static constexpr vx_size BATCH_SIZE = 2U;
static constexpr vx_uint8 INPUT_GRAPH_PARAM_INDEX = 0U;
static constexpr vx_uint8 OUTPUT_GRAPH_PARAM_INDEX = 1U;

void signal_handler(int signal)
{
    isExitRequested = true;
}

int main()
{
    std::signal(SIGINT, signal_handler);

    appInit();
    tivxInit();
    vx_context context;
    if (context = vxCreateContext(); vxGetStatus(reinterpret_cast<vx_reference>(context)) != VX_SUCCESS)
    {
        std::cout << "Failed to create context" << std::endl;
    }

    vx_image listOfInputImages[BATCH_SIZE];
    for (auto &input : listOfInputImages)
    {
        if (input = vxCreateImage(context, 1280, 960, VX_DF_IMAGE_U8); vxGetStatus(reinterpret_cast<vx_reference>(input)) != VX_SUCCESS)
        {
            std::cout << "Failed to create input image" << std::endl;
        }
    }

    vx_image listOfOutputImages[BATCH_SIZE];
    for (auto &output : listOfOutputImages)
    {
        if (output = vxCreateImage(context, 1280, 960, VX_DF_IMAGE_U8); vxGetStatus(reinterpret_cast<vx_reference>(output)) != VX_SUCCESS)
        {
            std::cout << "Failed to create output image" << std::endl;
        }
    }

    vx_graph graph;
    if (graph = vxCreateGraph(context); vxGetStatus(reinterpret_cast<vx_reference>(graph)) != VX_SUCCESS)
    {
        std::cout << "Failed to create graph" << std::endl;
    }

    vx_node scale_node;
    if (scale_node = vxScaleImageNode(graph, listOfInputImages[0], listOfOutputImages[0], VX_INTERPOLATION_BILINEAR); vxGetStatus(reinterpret_cast<vx_reference>(scale_node)) != VX_SUCCESS)
    {
        std::cout << "Failed to create scale node" << std::endl;
        std::runtime_error("Failed to create scale node");
    }

    if (auto result = vxSetNodeTarget(scale_node, VX_TARGET_STRING, TIVX_TARGET_VPAC_MSC1); result != VX_SUCCESS)
    {
        std::cout << "Failed to set node targets" << std::endl;
        std::runtime_error("Failed to set node targets");
    }

    vx_uint32 num_params;
    vxQueryNode(scale_node, VX_NODE_PARAMETERS, &num_params, sizeof(num_params));
    std::cout << "Number of parameters in scale_node: " << num_params << "\n";

    std::vector<vx_parameter> parameters;
    parameters.push_back(vxGetParameterByIndex(scale_node, 0));
    parameters.push_back(vxGetParameterByIndex(scale_node, 1));

    for (auto param : parameters)
    {
        vx_enum type;
        vxQueryParameter(param, VX_PARAMETER_TYPE, &type, sizeof(type));
        std::cout << "Parameter type: 0x" << std::hex << type << "\n";
    }

    for (auto param : parameters)
    {
        vxAddParameterToGraph(graph, param);
    }

    for (auto i = 0U; i < BATCH_SIZE; i++)
    {
        vxSetGraphParameterByIndex(graph, INPUT_GRAPH_PARAM_INDEX, reinterpret_cast<vx_reference>(listOfInputImages[i]));
        vxSetGraphParameterByIndex(graph, OUTPUT_GRAPH_PARAM_INDEX, reinterpret_cast<vx_reference>(listOfOutputImages[i]));
    }

    vx_graph_parameter_queue_params_t graph_parameters_queue_params_list[2U];

    graph_parameters_queue_params_list[0U].graph_parameter_index = INPUT_GRAPH_PARAM_INDEX;
    graph_parameters_queue_params_list[0U].refs_list_size = BATCH_SIZE;
    graph_parameters_queue_params_list[0U].refs_list = reinterpret_cast<vx_reference *>(&listOfInputImages[0U]);

    graph_parameters_queue_params_list[1U].graph_parameter_index = OUTPUT_GRAPH_PARAM_INDEX;
    graph_parameters_queue_params_list[1U].refs_list_size = BATCH_SIZE;
    graph_parameters_queue_params_list[1U].refs_list = reinterpret_cast<vx_reference *>(&listOfOutputImages[0U]);

    // Schedule config telling which paramters to enqueue to
    vxSetGraphScheduleConfig(
        graph,
        VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL,
        BATCH_SIZE,
        graph_parameters_queue_params_list);

    // Verify graph
    if (vxVerifyGraph(graph) != VX_SUCCESS)
    {
        std::cout << "Failed to verify graph" << std::endl;
        std::runtime_error("Failed to verify graph");
    }

    // Pipe up stage of the graph
    // Enqueue input/output image buffers
    vxGraphParameterEnqueueReadyRef(graph, INPUT_GRAPH_PARAM_INDEX, reinterpret_cast<vx_reference *>(&listOfInputImages[0]), 1);
    vxGraphParameterEnqueueReadyRef(graph, OUTPUT_GRAPH_PARAM_INDEX, reinterpret_cast<vx_reference *>(&listOfOutputImages[0]), 1);

    // Dequeue processed images
    vx_image input_done, output_done;

    // Processing of the graph
    while (!isExitRequested)
    {
        // SCHEDULE THE GRAPH
        if (vxScheduleGraph(graph) != VX_SUCCESS)
        {
            std::cout << "Failed to schedule graph" << std::endl;
            std::runtime_error("Failed to schedule graph");
        }

        vxWaitGraph(graph);

        vxGraphParameterDequeueDoneRef(graph, INPUT_GRAPH_PARAM_INDEX, reinterpret_cast<vx_reference *>(&input_done), 1, NULL);
        vxGraphParameterDequeueDoneRef(graph, OUTPUT_GRAPH_PARAM_INDEX, reinterpret_cast<vx_reference *>(&output_done), 1, NULL);

        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        std::cout << "Processing done" << std::endl;

        // Enqueue images back for further processing if needed
        vxGraphParameterEnqueueReadyRef(graph, INPUT_GRAPH_PARAM_INDEX, reinterpret_cast<vx_reference *>(&input_done), 1);
        vxGraphParameterEnqueueReadyRef(graph, OUTPUT_GRAPH_PARAM_INDEX, reinterpret_cast<vx_reference *>(&output_done), 1);
    }

    vxReleaseNode(&scale_node);

    for (auto input : listOfInputImages)
        vxReleaseImage(&input);

    for (auto output : listOfOutputImages)
        vxReleaseImage(&output);

    vxReleaseGraph(&graph);
    vxReleaseContext(&context);

    tivxDeInit();
    appDeInit();
}
