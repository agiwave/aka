#include <torch/extension.h>
#include <vector>

#ifndef __SHAPE_T__
#define __SHAPE_T__
typedef struct{
    int b, l, d, stepb, stepl;
}shape_t;
#endif//__SHAPE_T__
typedef struct{
    int x, y;
}INDICS;

#ifndef IDX
#define IDX_SCALE(shape) ((blockIdx.x % shape.b) * shape.stepb + blockIdx.y % shape.d)
#define IDX(shape) (blockIdx.x * shape.stepb + blockIdx.y)
#endif//IDX

#define atomAdd(p,b) ((*(p)) += (b))

namespace { namespace device {
    template <typename scalar_t> void causalScan_cpu_Forward(
        const shape_t shapeA,
        const shape_t shapeO,
        const shape_t shapeZ,
        scalar_t * pZ,
        scalar_t * pA,
        scalar_t * pX,
        scalar_t * pO,
        INDICS& blockIdx
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

    template <typename scalar_t> void causalScan_cpu_Backward(
        const shape_t shapeA,
        const shape_t shapeO,
        const shape_t shapeZ,
        scalar_t * gradZ,
        scalar_t * gradA,
        scalar_t * gradX,
        scalar_t * gradO,
        scalar_t * pZ,
        scalar_t * pA,
        scalar_t * pO,
        INDICS& blockIdx
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

#define SHAPE4D(t) {(int)t.size(0), (int)t.size(1), (int)t.size(2), (int)(t.size(1)*t.size(2)), (int)t.size(2) }

torch::Tensor causalScan_cpu_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B) {
    auto O = torch::zeros_like(B);
    shape_t shapeA = SHAPE4D(A);
    shape_t shapeO = SHAPE4D(B);
    shape_t shapeZ = SHAPE4D(Z);
    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan_Backward", ([&] {
        for(int ib=0; ib<shapeO.l; ib++)
        for(int ih=0; ih<shapeO.d; ih++){
            INDICS indics[] = {
                {ib, ih}
            };
            device::causalScan_cpu_Forward<scalar_t>(
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
    auto gradA = torch::zeros_like(A);
    auto gradX = torch::zeros_like(O);
    auto gradZ = torch::zeros_like(Z);
    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan_Backward", ([&] {
        shape_t shapeA = SHAPE4D(gradA);
        shape_t shapeO = SHAPE4D(gradO);
        shape_t shapeZ = SHAPE4D(gradZ);
        for(int ib=0; ib<shapeO.l; ib++)
        for(int ih=0; ih<shapeO.d; ih++){
            INDICS indics[] = {
                {ib, ih}
            };
            device::causalScan_cpu_Backward(
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

#ifndef __PYBINDED__
#define __PYBINDED__
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan_cpu_Forward, "");
    m.def("backward", &causalScan_cpu_Backward, "");
}
#endif//__PYBINDED__