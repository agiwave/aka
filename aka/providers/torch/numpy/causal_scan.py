import torch
import torch.cuda as cuda
import torch.utils.cpp_extension as ext
import os

script_dir = os.path.dirname(__file__)
if cuda.is_available():
    causal_scan_kernel = ext.load('causal_scan', [
        os.path.join(script_dir, 'causal_scan_k_cpu.cpp'),
        os.path.join(script_dir, 'causal_scan_k_cuda.cu'),
        os.path.join(script_dir, 'causal_scan_cuda.cpp')
    ]) 
else:
    causal_scan_kernel = ext.load('causal_scan', [
        os.path.join(script_dir, 'causal_scan_k_cpu.cpp'),
        os.path.join(script_dir, 'causal_scan_cpu.cpp')
    ]) 

class causal_scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Z, A, B):
        bz, lz, hz, dz = Z.shape
        ba, la, ha, da = A.shape
        bb, lb, hb, db = B.shape
        assert lz == 1 and la == lb
        assert bz == bb
        assert ba == 1 or ba == bb
        assert hz == hb and dz == db
        assert ha == hb
        assert da == 1 or da == db 
        O = causal_scan_kernel.forward(Z, A, B)
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
    print(causal_scan.apply(Z, A, B))
