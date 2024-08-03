#include "hello_world_tda4.h"
#include <vector>
#include <string>
#include <iostream>

#include "TI/tivx.h"
#include "processor_sdk/vision_apps/utils/app_init/include/app_init.h"
int main()
{
    appInit();
    tivxInit();

    vx_context context = vxCreateContext();
    vx_graph graph = vxCreateGraph(context);

    if (auto result = vxVerifyGraph(graph); result != VX_SUCCESS)
    {
        std::cout << "Failed to verify graph" << std::endl;
        return 1;
    }

    tivxDeInit();
    appDeInit();
}
