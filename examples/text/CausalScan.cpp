#include <torch/extension.h>
#include <vector>

typedef struct{
    int x, y;
}INDICS;

#define causalScan_Forward_cuda causalScan_Forward_cpu
#define causalScan_Backward_cuda causalScan_Backward_cpu
#define DEVICEINDICS ,const INDICS& blockIdx
#define __global__
#define atomicAdd(p,b) (*(p) = *(p) + (b))
#define __INLINE_CPP__
#include "CausalScan.cu"

#ifndef __DISABLE_CUDA__
torch::Tensor causalScan_cuda_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B);
std::vector<torch::Tensor> causalScan_cuda_Backward(torch::Tensor gradO, torch::Tensor Z, torch::Tensor A, torch::Tensor O);
#endif//

torch::Tensor causalScan_cpu_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B) {
#ifndef __DISABLE_CUDA__
    if(A.is_cuda()) {
        return causalScan_cuda_Forward(Z,A,B);
    }
#endif
    auto O = torch::zeros_like(B);
    shape_t shapeA = SHAPE4D(A);
    shape_t shapeO = SHAPE4D(B);
    shape_t shapeZ = SHAPE4D(Z);
    AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Backward", ([&] {
        for(int ib=0; ib<shapeO.l; ib++)
        for(int ih=0; ih<shapeO.d; ih++){
            INDICS indics[] = {
                {ib, ih}
            };
            device::causalScan_Forward_cpu<scalar_t>(
                shapeA,
                shapeO,
                shapeZ,
                (scalar_t*)Z.data_ptr(),
                (scalar_t*)A.data_ptr(),
                (scalar_t*)B.data_ptr(),
                (scalar_t*)O.data_ptr(),
                indics[0]
            );
        }
    }));
    return O;
}

std::vector<torch::Tensor> causalScan_cpu_Backward(torch::Tensor gradO, torch::Tensor Z, torch::Tensor A, torch::Tensor O) {
#ifndef __DISABLE_CUDA__
    if(A.is_cuda()) {
        return causalScan_cuda_Backward(gradO,Z,A,O);
    }
#endif
    auto gradA = torch::zeros_like(A);
    auto gradX = torch::zeros_like(O);
    auto gradZ = torch::zeros_like(Z);
    AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Backward", ([&] {
        shape_t shapeA = SHAPE4D(gradA);
        shape_t shapeO = SHAPE4D(gradO);
        shape_t shapeZ = SHAPE4D(gradZ);
        for(int ib=0; ib<shapeO.l; ib++)
        for(int ih=0; ih<shapeO.d; ih++){
            INDICS indics[] = {
                {ib, ih}
            };
            device::causalScan_Backward_cpu(
                shapeA,
                shapeO,
                shapeZ,
                (scalar_t*)gradZ.data_ptr(),
                (scalar_t*)gradA.data_ptr(),
                (scalar_t*)gradX.data_ptr(),
                (scalar_t*)gradO.data_ptr(),
                (scalar_t*)Z.data_ptr(),
                (scalar_t*)A.data_ptr(),
                (scalar_t*)O.data_ptr(),
                indics[0]
            );
        }
    }));
    return {gradZ, gradA, gradX};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan_cpu_Forward, "");
    m.def("backward", &causalScan_cpu_Backward, "");
}
