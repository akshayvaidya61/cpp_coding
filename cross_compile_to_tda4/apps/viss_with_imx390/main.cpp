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
#include "vision_apps/utils/app_init/include/app_init.h"
#include "imaging/kernels/include/TI/hwa_vpac_viss.h"
#include "imaging/sensor_drv/include/iss_sensors.h"
#include "imaging/kernels/include/TI/hwa_kernels.h"
#include "tiovx/utils/include/tivx_utils_file_rd_wr.h"
#include "imaging/utils/iss/include/app_iss.h"
#include "app_utils/utils/mem/include/app_mem.h"
#include "TI/tivx_log_rt.h"
}

bool isExitRequested = false;
void signal_handler(int signal)
{
    isExitRequested = true;
}

vx_node const vxCreateVissWithImx390Node(vx_graph, vx_context);

int main()
{

    std::signal(SIGINT, signal_handler);

    std::string filename;

    appInit();
    tivxInit();

    vx_context context;
    if (context = vxCreateContext(); vxGetStatus(reinterpret_cast<vx_reference>(context)) != VX_SUCCESS)
    {
        std::cout << "Failed to create context" << std::endl;
    }

    vx_graph graph;
    if (graph = vxCreateGraph(context); vxGetStatus(reinterpret_cast<vx_reference>(graph)) != VX_SUCCESS)
    {
        std::cout << "Failed to create graph" << std::endl;
    }

    vx_node viss_node;
    if (viss_node = vxCreateVissWithImx390Node(graph, context); vxGetStatus(reinterpret_cast<vx_reference>(viss_node)) != VX_SUCCESS)
    {
        std::cout << "Failed to create viss node" << std::endl;
    }

    vx_image input;
    if (input = vxCreateImage(context, 1280, 960, VX_DF_IMAGE_U8); vxGetStatus(reinterpret_cast<vx_reference>(input)) != VX_SUCCESS)
    {
        std::cout << "Failed to create input image" << std::endl;
        std::runtime_error("Failed to create input image");
    }

    tivxDeInit();
    appDeInit();
}

vx_user_data_object createDccParamViss(vx_context context, const char *sensor_name, uint32_t sensor_dcc_mode)
{
    const vx_char dcc_viss_user_data_object_name[] = "dcc_viss";
    vx_size dcc_buff_size = appIssGetDCCSizeVISS(const_cast<char *>(sensor_name), sensor_dcc_mode);
    if (dcc_buff_size == 0)
    {
        printf("Incorrect DCC Buf size \n");
        return NULL;
    }

    vx_user_data_object dcc_param_viss = vxCreateUserDataObject(context, dcc_viss_user_data_object_name, dcc_buff_size, NULL);

    vx_map_id dcc_viss_buf_map_id;
    uint8_t *dcc_viss_buf;
    vxMapUserDataObject(dcc_param_viss, 0, dcc_buff_size, &dcc_viss_buf_map_id, (void **)&dcc_viss_buf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    memset(dcc_viss_buf, 0xAB, dcc_buff_size);

    int32_t dcc_status = appIssGetDCCBuffVISS(const_cast<char *>(sensor_name), sensor_dcc_mode, dcc_viss_buf, dcc_buff_size);
    if (dcc_status != 0)
    {
        printf("GetDCCBuf failed\n");
        vxUnmapUserDataObject(dcc_param_viss, dcc_viss_buf_map_id);
        return NULL;
    }

    vxUnmapUserDataObject(dcc_param_viss, dcc_viss_buf_map_id);
    return dcc_param_viss;
}

static constexpr uint32_t AE_GAIN_MAGIC_VALUE = 1030;
static constexpr uint32_t AE_EXPOSURE_TIME_MAGIC_VALUE = 16666;
static constexpr uint32_t AWB_TEMPERATURE_MAGIC_VALUE = 3000;
static constexpr uint32_t AWB_GAIN_MAGIC_VALUE = 525;
static constexpr uint32_t AWB_OFFSET_MAGIC_VALUE = 2;

vx_user_data_object createAeAwbResult(vx_context context)
{
    tivx_ae_awb_params_t ae_awb_params;
    tivx_ae_awb_params_init(&ae_awb_params);

    ae_awb_params.ae_valid = 1;
    ae_awb_params.exposure_time = AE_EXPOSURE_TIME_MAGIC_VALUE;
    ae_awb_params.analog_gain = AE_GAIN_MAGIC_VALUE;
    ae_awb_params.awb_valid = 1;
    ae_awb_params.color_temperature = AWB_TEMPERATURE_MAGIC_VALUE;
    for (int i = 0; i < 4; i++)
    {
        ae_awb_params.wb_gains[i] = AWB_GAIN_MAGIC_VALUE;
        ae_awb_params.wb_offsets[i] = AWB_OFFSET_MAGIC_VALUE;
    }

    return vxCreateUserDataObject(context, "tivx_ae_awb_params_t", sizeof(tivx_ae_awb_params_t), &ae_awb_params);
}

static constexpr uint32_t META_HEIGHT_AFTER = 4;
static constexpr uint32_t NUM_EXPOSURES = 1;
static constexpr uint32_t MSB_16_BIT = 11;
static constexpr uint32_t META_HEIGHT_BEFORE = 0;

tivx_raw_image createRawImage(vx_context context, vx_uint32 width, vx_uint32 height)
{
    tivx_raw_image_create_params_t raw_params;

    raw_params.width = width;
    raw_params.height = height;
    raw_params.meta_height_after = META_HEIGHT_AFTER;
    raw_params.num_exposures = NUM_EXPOSURES;
    raw_params.line_interleaved = vx_false_e;
    raw_params.format[0].pixel_container = TIVX_RAW_IMAGE_16_BIT;
    raw_params.format[0].msb = MSB_16_BIT;
    raw_params.format[1].pixel_container = TIVX_RAW_IMAGE_16_BIT;
    raw_params.format[1].msb = MSB_16_BIT;
    raw_params.format[2].pixel_container = TIVX_RAW_IMAGE_16_BIT;
    raw_params.format[2].msb = MSB_16_BIT;
    raw_params.meta_height_before = META_HEIGHT_BEFORE;

    return tivxCreateRawImage(context, &raw_params);
}

static constexpr const char *H3A_DATA_OBJECT_NAME = "tivx_h3a_data_t";
static constexpr size_t H3A_DATA_OBJECT_SIZE = sizeof(tivx_h3a_data_t);

vx_user_data_object createH3aAewAf(vx_context context)
{
    vx_user_data_object h3a_aew_af = vxCreateUserDataObject(context, H3A_DATA_OBJECT_NAME, H3A_DATA_OBJECT_SIZE, nullptr);

    if (h3a_aew_af != nullptr)
    {
        vx_map_id map_id;
        uint8_t *buffer;
        vxMapUserDataObject(h3a_aew_af, 0, H3A_DATA_OBJECT_SIZE, &map_id, reinterpret_cast<void **>(&buffer), VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
        memset(buffer, 0, H3A_DATA_OBJECT_SIZE);
        vxUnmapUserDataObject(h3a_aew_af, map_id);
    }

    return h3a_aew_af;
}

static constexpr const char *CONFIGURATION_OBJECT_NAME = "tivx_vpac_viss_params_t";
static constexpr size_t CONFIGURATION_OBJECT_SIZE = sizeof(tivx_vpac_viss_params_t);
static constexpr const char *AE_AWB_RESULT_OBJECT_NAME = "tivx_ae_awb_params_t";
static constexpr size_t AE_AWB_RESULT_OBJECT_SIZE = sizeof(tivx_ae_awb_params_t);
static constexpr const char *DCC_PARAM_VISS_OBJECT_NAME = "dcc_viss";

vx_node createVissNode(vx_graph graph, vx_context context, const tivx_vpac_viss_params_t *params, tivx_raw_image raw, vx_image y8_r8_c2)
{
    vx_user_data_object configuration = vxCreateUserDataObject(context, CONFIGURATION_OBJECT_NAME, CONFIGURATION_OBJECT_SIZE, params);
    vx_user_data_object ae_awb_result = createAeAwbResult(context);
    vx_user_data_object dcc_param_viss = createDccParamViss(context, SENSOR_SONY_IMX390_UB953_D3, 0);
    vx_user_data_object h3a_aew_af = createH3aAewAf(context);

    return tivxVpacVissNode(graph, configuration, ae_awb_result, dcc_param_viss, raw, NULL, NULL, y8_r8_c2, NULL, NULL, h3a_aew_af, NULL, NULL, NULL);
}

static constexpr auto VISS_TARGET = TIVX_TARGET_VPAC_VISS1;
static constexpr auto VISS_H3A_IN_LSC = TIVX_VPAC_VISS_H3A_IN_LSC;
static constexpr auto VISS_H3A_MODE_AEWB = TIVX_VPAC_VISS_H3A_MODE_AEWB;
static constexpr auto VISS_SENSOR_DCC_ID_390 = 390;
static constexpr auto VISS_EE_MODE_OFF = TIVX_VPAC_VISS_EE_MODE_OFF;
static constexpr auto VISS_MUX2_NV12 = TIVX_VPAC_VISS_MUX2_NV12;
static constexpr auto VISS_CHROMA_MODE_420 = TIVX_VPAC_VISS_CHROMA_MODE_420;
static constexpr auto VISS_MUX2_YUV422 = TIVX_VPAC_VISS_MUX2_YUV422;
static constexpr auto VISS_CHROMA_MODE_422 = TIVX_VPAC_VISS_CHROMA_MODE_422;
static constexpr auto RAW_IMG_WIDTH = 1936;
static constexpr auto RAW_IMG_HEIGHT = 1096;

vx_node const vxCreateVissWithImx390Node(vx_graph currentGraph, vx_context currentContext)
{
    vx_node node = nullptr;
    if (vx_true_e == tivxIsTargetEnabled(VISS_TARGET))
    {
        tivxHwaLoadKernels(currentContext);

        tivx_raw_image raw = createRawImage(currentContext, RAW_IMG_WIDTH, RAW_IMG_HEIGHT);
        vx_uint32 width, height;
        tivxQueryRawImage(raw, TIVX_RAW_IMAGE_WIDTH, &width, sizeof(width));
        tivxQueryRawImage(raw, TIVX_RAW_IMAGE_HEIGHT, &height, sizeof(height));

        vx_image y8_r8_c2 = vxCreateImage(currentContext, width, height, VX_DF_IMAGE_NV12);

        tivx_vpac_viss_params_t params;
        tivx_vpac_viss_params_init(&params);
        params.h3a_in = VISS_H3A_IN_LSC;
        params.h3a_aewb_af_mode = VISS_H3A_MODE_AEWB;
        params.sensor_dcc_id = VISS_SENSOR_DCC_ID_390;
        params.bypass_glbce = 0;
        params.bypass_nsf4 = 0;
        params.fcp[0].ee_mode = VISS_EE_MODE_OFF;
        params.fcp[0].mux_output2 = VISS_MUX2_NV12;
        params.fcp[0].chroma_mode = VISS_CHROMA_MODE_420;
        params.fcp[1].ee_mode = VISS_EE_MODE_OFF;
        params.fcp[1].mux_output2 = VISS_MUX2_YUV422;
        params.fcp[1].chroma_mode = VISS_CHROMA_MODE_422;

        node = createVissNode(currentGraph, currentContext, &params, raw, y8_r8_c2);

        vxSetNodeTarget(node, VX_TARGET_STRING, VISS_TARGET);
    }
    return node;
}
