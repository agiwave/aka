#include <torch/extension.h>
#include <vector>

#ifndef __SHAPE_T__
#define __SHAPE_T__
typedef struct{
    int x, y, z, l, s;
}shape_t;
#endif//__SHAPE_T__
typedef struct{
    int x, y, z;
}INDICS;

#ifndef IDX
#define IDX(shape) (((blockIdx.x % shape.x) * shape.l * shape.y + blockIdx.y % shape.y ) * shape.z + blockIdx.z % shape.z)
#endif//IDX

#define atomAdd(p,b) ((*(p)) += (b))

namespace { namespace device {
    template <typename scalar_t> void causalScan4d_cpu_Forward(
        int length,
        const shape_t shapeA,
        const shape_t shapeB,
        const shape_t shapeZ,
        scalar_t * Z,
        scalar_t * A,
        scalar_t * B,
        scalar_t * O,
        INDICS& blockIdx
    )
    {
        int idx_A = IDX(shapeA);
        int idx_B = IDX(shapeB);
        int idx_Z = IDX(shapeZ);
        scalar_t zh = Z[idx_Z];
        while(length-->0) {
            zh = A[idx_A] * zh + B[idx_B];
            O[idx_B] = zh;
            idx_A += shapeA.s;
            idx_B += shapeB.s;
        }
    }

    template <typename scalar_t> void causalScan4d_cpu_Backward(
        int length,
        const shape_t shapeA,
        const shape_t shapeB,
        const shape_t shapeZ,
        scalar_t * gradZ,
        scalar_t * gradA,
        scalar_t * gradB,
        scalar_t * gradO,
        scalar_t * Z,
        scalar_t * A,
        scalar_t * O,
        INDICS& blockIdx
    )
    {
        int idx_A = IDX(shapeA);
        int idx_B = IDX(shapeB);
        int idx_Z = IDX(shapeZ);
        idx_A += shapeA.s * (length - 1);
        idx_B += shapeA.s * (length - 1);
        scalar_t grad = 0.0;
        while(length-->1) {
            grad += gradO[idx_B];
            gradB[idx_B] = grad;
            atomAdd(gradA + idx_A, O[idx_B - shapeB.s] * grad);
            grad *= A[idx_A];
            idx_A -= shapeA.s;
            idx_B -= shapeB.s;
        }
        grad += gradO[idx_B];
        gradB[idx_B] = grad;
        atomAdd(gradA + idx_A, Z[idx_Z] * grad);
        gradZ[idx_Z] = A[idx_A] * grad;
    }
}}


#define SHAPE4D(t) {(int)t.size(0), (int)t.size(2), (int)t.size(3), (int)t.size(1), (int)t.size(2) * (int)t.size(3) }

torch::Tensor causalScan4d_cpu_Forward(torch::Tensor Z, torch::Tensor A, torch::Tensor B) {
    auto O = torch::zeros_like(B);
    shape_t shapeA = SHAPE4D(A);
    shape_t shapeB = SHAPE4D(B);
    shape_t shapeZ = SHAPE4D(Z);

    int b = O.size(0);
    int length = O.size(1);
    int h = O.size(2);
    int d = O.size(3);
    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan4d_Backward", ([&] {
        for(int ib=0; ib<b; ib++)
        for(int ih=0; ih<h; ih++)
        for(int id=0; id<d; id++){
            INDICS indics[] = {
                {ib, ih, id}
            };
            device::causalScan4d_cpu_Forward<scalar_t>(
                length,
                shapeA,
                shapeB,
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

std::vector<torch::Tensor> causalScan4d_cpu_Backward(torch::Tensor gradO, torch::Tensor Z, torch::Tensor A, torch::Tensor O) {
    auto gradA = torch::zeros_like(A);
    auto gradB = torch::zeros_like(O);
    auto gradZ = torch::zeros_like(O.slice(1, 0, 1));
    shape_t shapeA = SHAPE4D(gradA);
    shape_t shapeB = SHAPE4D(gradB);
    shape_t shapeZ = SHAPE4D(gradZ);

    int b = O.size(0);
    int length = O.size(1);
    int h = O.size(2);
    int d = O.size(3);
    AT_DISPATCH_FLOATING_TYPES(O.type(), "causalScan4d_Backward", ([&] {
        for(int ib=0; ib<b; ib++)
        for(int ih=0; ih<h; ih++)
        for(int id=0; id<d; id++){
            INDICS indics[] = {
                {ib, ih, id}
            };
            device::causalScan4d_cpu_Backward(
                length,
                shapeA,
                shapeB,
                shapeZ,
                (scalar_t*)gradZ.data_ptr(),
                (scalar_t*)gradA.data_ptr(),
                (scalar_t*)gradB.data_ptr(),
                (scalar_t*)gradO.data_ptr(),
                (scalar_t*)Z.data_ptr(),
                (scalar_t*)A.data_ptr(),
                (scalar_t*)O.data_ptr(),
                indics[0]
            );
        }
    }));
    return {gradZ, gradA, gradB};
}

#ifndef __PYBINDED__
#define __PYBINDED__
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &causalScan4d_cpu_Forward, "");
    m.def("backward", &causalScan4d_cpu_Backward, "");
}
#endif//__PYBINDED__