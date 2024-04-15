#include <cuda.h>
#include <cuda_runtime.h>

#ifndef __SHAPE_T__
#define __SHAPE_T__
typedef struct{
    int b, l, d, stepb, stepl;
}shape_t;
#endif//__SHAPE_T__

#ifndef IDX
#define IDX_SCALE(shape) ((blockIdx.x % shape.b) * shape.stepb + blockIdx.y % shape.d)
#define IDX(shape) (blockIdx.x * shape.stepb + blockIdx.y)
#endif//IDX

#define atomAdd atomicAdd

namespace { namespace device {
    template <typename scalar_t> __global__ void causalScan_Forward(
        const shape_t shapeA,
        const shape_t shapeO,
        const shape_t shapeZ,
        scalar_t * pZ,
        scalar_t * pA,
        scalar_t * pX,
        scalar_t * pO
    )
    {
        int idxX = IDX(shapeO);
        pA += IDX_SCALE(shapeA);
        pX += idxX;
        pO += idxX;
        scalar_t zh = pZ[IDX(shapeZ)];
        int length = shapeO.l;
        while(length-->0) {
            zh = (*pA) * zh + (*pX);
            (*pO) = zh;
            pA += shapeA.stepl;
            pX += shapeO.stepl;
            pO += shapeO.stepl;
        }
    }

    template <typename scalar_t> __global__ void causalScan_Backward(
        const shape_t shapeA,
        const shape_t shapeO,
        const shape_t shapeZ,
        scalar_t * gradZ,
        scalar_t * gradA,
        scalar_t * gradX,
        scalar_t * gradO,
        scalar_t * pZ,
        scalar_t * pA,
        scalar_t * pO
    )
    {
        scalar_t grad = 0.0;
        int idxA = IDX_SCALE(shapeA);
        int idxO = IDX(shapeO);
        int idxZ = IDX(shapeZ);
        int length = shapeO.l;
        int offsetA = idxA + shapeA.stepl * (length - 1);
        int offsetO = idxO + shapeO.stepl * (length - 1);
        pA += offsetA;
        pO += offsetO-shapeO.stepl; // h(n-1)
        gradA += offsetA;
        gradX += offsetO;
        gradO += offsetO;
        while(length-->1) {
            grad += *gradO;
            (*gradX) = grad;
            atomAdd(gradA, (*pO) * grad);
            grad *= (*pA);
            gradA -= shapeA.stepl;
            gradX -= shapeO.stepl;
            gradO -= shapeO.stepl;
            pA -= shapeA.stepl;
            pO -= shapeO.stepl;
        }
        grad += *gradO;
        (*gradX) = grad;
        atomAdd(gradA, pZ[idxZ] * grad);
        gradZ[idxZ] = (*pA) * grad;
    }
}}

#undef atomAdd
#define __PYBINDED__
#include "./CausalScan.hpp"
torch::Tensor causalScan_cuda_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B) {
    if(!A.is_cuda()) {
        return causalScan_cpu_Forward(Z,A,B);
    }
    auto O = torch::zeros_like(B);
    AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Forward", ([&] {
        shape_t shapeA = SHAPE4D(A);
        shape_t shapeO = SHAPE4D(B);
        shape_t shapeZ = SHAPE4D(Z);
        const dim3 blocks(O.size(0), O.size(2));
        device::causalScan_Forward<scalar_t><<<blocks, 1>>>(
            shapeA,
            shapeO,
            shapeZ,
            (scalar_t*)Z.data_ptr(),
            (scalar_t*)A.data_ptr(),
            (scalar_t*)B.data_ptr(),
            (scalar_t*)O.data_ptr()
        );
    }));
    return O;
}

std::vector<torch::Tensor> causalScan_cuda_Backward(torch::Tensor gradO, torch::Tensor Z, torch::Tensor A, torch::Tensor O) {
    if(!A.is_cuda()) {
        return causalScan_cpu_Backward(gradO,Z,A,O);
    }
    auto gradA = torch::zeros_like(A);
    auto gradX = torch::zeros_like(O);
    auto gradZ = torch::zeros_like(Z);
    AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Backward", ([&] {
        shape_t shapeA = SHAPE4D(gradA);
        shape_t shapeO = SHAPE4D(gradX);
        shape_t shapeZ = SHAPE4D(gradZ);
        const dim3 blocks(O.size(0), O.size(2));
        device::causalScan_Backward<scalar_t><<<blocks, 1>>>(
            shapeA,
            shapeO,
            shapeZ,
            (scalar_t*)gradZ.data_ptr(),
            (scalar_t*)gradA.data_ptr(),
            (scalar_t*)gradX.data_ptr(),
            (scalar_t*)gradO.data_ptr(),
            (scalar_t*)Z.data_ptr(),
            (scalar_t*)A.data_ptr(),
            (scalar_t*)O.data_ptr()
        );
    }));
    return {gradZ, gradA, gradX};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan_cuda_Forward, "");
    m.def("backward", &causalScan_cuda_Backward, "");
}