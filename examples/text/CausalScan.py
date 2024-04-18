try:
    import torch.cuda as cuda
    import torch.utils.cpp_extension as ext
    import os
    script_dir = os.path.dirname(__file__)
    causal_scan_kernel = ext.load('extCausalScan', [
        os.path.join(script_dir, 'CausalScan.' + ('cu' if cuda.is_available() else 'cpp'))
    ])
except ImportError:
    causal_scan_kernel = None
    print('Warn: CausalScan4d import failed.')

import torch
class CausalScan(torch.autograd.Function):
    '''
    Formula:
    h(1) = a(1) * z + b(1)
    h(2) = a(2) * h(1) + b(2)
    ...
    h(n) = a(n) * h(n-1) + b(n)

    Args:
    h : (b, 1, d)
    A : (b, l, d)
    x : (b, l, d)

    Return:
    y : (b, l, d)
    '''
    @staticmethod
    def forward(ctx, x, h, A):
        for item in [x, h, A]:
            assert len(item.shape) == 3
            assert item.size(0) == 1 or item.size(0) == h.size(0)
            assert item.size(1) == 1 or item.size(1) == x.size(1)
            assert h.size(2) % item.size(2) == 0

        assert h.size(1) == 1
        x = x.contiguous()
        h = h.contiguous()
        A = A.contiguous()
        x = causal_scan_kernel.forward(x, h, A)
        ctx.save_for_backward(x, h, A)
        return x

    @staticmethod
    def backward(ctx, gradO):
        x, h, A = ctx.saved_variables
        gradx, gradh, gradA = causal_scan_kernel.backward(gradO, x, h, A)
        return gradx, gradh, gradA

if __name__ == "__main__":
    device = torch.device("cuda")
    Z = torch.randn(5, 1, 3, device=device)
    A = torch.randn(5, 2, 3, device=device)
    X = torch.randn(5, 2, 3, device=device)
    print(CausalScan.apply(X, Z, A))

causalScan = None if causal_scan_kernel is None else CausalScan.apply