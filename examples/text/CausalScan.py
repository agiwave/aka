import torch
import torch.cuda as cuda
import torch.utils.cpp_extension as ext
import os

script_dir = os.path.dirname(__file__)
if cuda.is_available():
    causal_scan_kernel = ext.load('extCausalScan', [
        os.path.join(script_dir, 'CausalScan.cu')
    ]) 
else:
    causal_scan_kernel = ext.load('extCausalScan', [
        os.path.join(script_dir, 'CausalScan.hpp')
    ]) 

class CausalScan(torch.autograd.Function):
    '''
    Formula:
    h(1) = a(1) * z + b(1)
    h(2) = a(2) * h(1) + b(2)
    ...
    h(n) = a(n) * h(n-1) + b(n)
    '''
    @staticmethod
    def forward(ctx, Z, A, B):
        bz, lz, dz = Z.shape
        ba, la, da = A.shape
        bb, lb, db = B.shape
        assert lz == 1 and lb % la == 0
        assert bz == bb
        assert ba == 1 or ba == bb
        assert dz == db 
        assert dz == db and db % da == 0
        O = causal_scan_kernel.forward(Z.contiguous(), A.contiguous(), B.contiguous())
        ctx.save_for_backward(Z, A, O)
        return O

    @staticmethod
    def backward(ctx, gradO):
        Z, A, O = ctx.saved_variables
        gradZ, gradA, gradB = causal_scan_kernel.backward(gradO, Z, A, O)
        return gradZ, gradA, gradB

if __name__ == "__main__":
    device = torch.device("cuda")
    Z = torch.randn(5, 1, 3, 4, device=device)
    A = torch.randn(5, 2, 3, 1, device=device)
    B = torch.randn(5, 2, 3, 4, device=device)
    print(CausalScan.apply(Z, A, B))
