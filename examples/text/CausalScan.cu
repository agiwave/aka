#ifndef __INLINE_CPP__
#define DEVICEINDICS 
#include <cuda.h>
#include <cuda_runtime.h>
#endif//__INLINE_CPP__

typedef struct{
    int b, l, d, stepb, stepl;
}shape_t;
#define SHAPE4D(t) {(int)t.size(0), (int)t.size(1), (int)t.size(2), \
    (int)(t.size(1)*t.size(2)), \
    (t.size(1)==1)?0:(int)t.size(2) \
}
#define IDX_SCALE(shape) ((blockIdx.x % shape.b) * shape.stepb + blockIdx.y % shape.d)
#define IDX(shape) (blockIdx.x * shape.stepb + blockIdx.y)

namespace { namespace device {
    template <typename scalar_t> __global__ void causalScan_Forward_cuda(
        const shape_t shapeA,
        const shape_t shapeO,
        const shape_t shapeZ,
        scalar_t * pZ,
        scalar_t * pA,
        scalar_t * pX,
        scalar_t * pO
        DEVICEINDICS
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

    template <typename scalar_t> __global__ void causalScan_Backward_cuda(
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
        DEVICEINDICS
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
            atomicAdd(gradA, (*pO) * grad);
            grad *= (*pA);
            gradA -= shapeA.stepl;
            gradX -= shapeO.stepl;
            gradO -= shapeO.stepl;
            pA -= shapeA.stepl;
            pO -= shapeO.stepl;
        }
        grad += *gradO;
        (*gradX) = grad;
        atomicAdd(gradA, pZ[idxZ] * grad);
        gradZ[idxZ] = (*pA) * grad;
    }
}}

#ifndef __INLINE_CPP__
#include <torch/extension.h>
#include <vector>
torch::Tensor causalScan_cuda_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B) {
    auto O = torch::zeros_like(B);
    AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Forward", ([&] {
        shape_t shapeA = SHAPE4D(A);
        shape_t shapeO = SHAPE4D(B);
        shape_t shapeZ = SHAPE4D(Z);
        const dim3 blocks(O.size(0), O.size(2));
        device::causalScan_Forward_cuda<scalar_t><<<blocks, 1>>>(
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
    auto gradA = torch::zeros_like(A);
    auto gradX = torch::zeros_like(O);
    auto gradZ = torch::zeros_like(Z);
    AT_DISPATCH_FLOATING_TYPES(O.scalar_type(), "causalScan_Backward", ([&] {
        shape_t shapeA = SHAPE4D(gradA);
        shape_t shapeO = SHAPE4D(gradX);
        shape_t shapeZ = SHAPE4D(gradZ);
        const dim3 blocks(O.size(0), O.size(2));
        device::causalScan_Backward_cuda<scalar_t><<<blocks, 1>>>(
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
#endif//__INLINE_CPP__