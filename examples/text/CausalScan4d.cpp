#include <torch/extension.h>
#include <vector>

typedef struct{
    int x, y;
}INDICS;
#define DEVICETYPE cpu
#define DEVICEINDICS ,const INDICS& blockIdx, const INDICS& threadIdx
#define __global__
#define atomicAdd(p,b) (*(p) = *(p) + (b))
#define __INLINE_CPP__
#include "CausalScan4d.cu"

#ifndef SHAPE5D
#define SHAPE5D(t) {\
    (int)t.size(0), (int)t.size(1), (int)t.size(2), (int)t.size(3), \
    (int)(t.size(1) * t.size(2) * t.size(3)),\
    (t.size(1) == 1) ? 0 : (int)(t.size(2) * t.size(3)),\
    (scalar_t*)t.data_ptr()\
}
#endif//SHAPE5D

#ifndef __DISABLE_CUDA__
torch::Tensor causalScan4d_cuda_Forward(
    torch::Tensor X, 
    torch::Tensor Z, 
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
);
std::vector<torch::Tensor> causalScan4d_cuda_Backward(
    torch::Tensor gradO,
    torch::Tensor X, 
    torch::Tensor Z,
    torch::Tensor A,
    torch::Tensor B, 
    torch::Tensor C
);
#endif//

torch::Tensor causalScan4d_Forward(
    torch::Tensor X, 
    torch::Tensor Z, 
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C
) {
#ifndef __DISABLE_CUDA__
    if(X.is_cuda()) {
        return causalScan4d_cuda_Forward(X,Z,A,B,C);
    }
#endif//

    auto O = torch::zeros_like(X);
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "causalScan4d_Forward", ([&] {
        wrap_t<scalar_t> shapeX = SHAPE5D(X);
        wrap_t<scalar_t> shapeZ = SHAPE5D(Z);
        wrap_t<scalar_t> shapeA = SHAPE5D(A);
        wrap_t<scalar_t> shapeB = SHAPE5D(B);
        wrap_t<scalar_t> shapeC = SHAPE5D(C);
        wrap_t<scalar_t> shapeO = SHAPE5D(O);
        for(int ib=0; ib<shapeZ.b; ib++)
        for(int ih=0; ih<shapeZ.d; ih++)
        for(int in=0; in<shapeZ.n; in++)
        {
            INDICS indics[] = {
                {ib, ih},
                {in}
            };
            device::causalScan4d_Forward_DEVICETYPE<scalar_t>(
                shapeX,
                shapeZ,
                shapeA,
                shapeB,
                shapeC,
                shapeO,
                indics[0],
                indics[1]
            );
        }
    }));
    return O;
}

std::vector<torch::Tensor> causalScan4d_Backward(
    torch::Tensor gradO,
    torch::Tensor X, 
    torch::Tensor Z,
    torch::Tensor A,
    torch::Tensor B, 
    torch::Tensor C
) {
#ifndef __DISABLE_CUDA__
    if(X.is_cuda()) {
        return causalScan4d_cuda_Backward(gradO,X,Z,A,B,C);
    }
#endif//
    auto gradX = torch::zeros_like(X);
    auto gradZ = torch::zeros_like(Z);
    auto gradA = torch::zeros_like(A);
    auto gradB = torch::zeros_like(B);
    auto gradC = torch::zeros_like(C);
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "causalScan4d_Backward", ([&] {
        wrap_t<scalar_t> deltaX = SHAPE5D(gradX);
        wrap_t<scalar_t> deltaO = SHAPE5D(gradO);
        wrap_t<scalar_t> deltaZ = SHAPE5D(gradZ);
        wrap_t<scalar_t> deltaA = SHAPE5D(gradA);
        wrap_t<scalar_t> deltaB = SHAPE5D(gradB);
        wrap_t<scalar_t> deltaC = SHAPE5D(gradC);
        for(int ib=0; ib<deltaZ.b; ib++)
        for(int ih=0; ih<deltaZ.d; ih++)
        for(int in=0; in<deltaZ.n; in++)
        {
            INDICS indics[] = {
                {ib, ih},
                {in}
            };
            device::causalScan4d_Backward_DEVICETYPE<scalar_t>(
                (scalar_t*)X.data_ptr(),
                (scalar_t*)Z.data_ptr(),
                (scalar_t*)A.data_ptr(),
                (scalar_t*)B.data_ptr(),
                (scalar_t*)C.data_ptr(),
                deltaO,
                deltaX,
                deltaZ,
                deltaA,
                deltaB,
                deltaC,
                indics[0],
                indics[1]
            );
        }
    }));
    return {gradX, gradZ, gradA, gradB, gradC};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan4d_Forward, "");
    m.def("backward", &causalScan4d_Backward, "");
}
