import aka.nn as nn
import aka.numpy as np

def SSMBlock(**kwargs):
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.num_heads = getattr(args, 'num_heads', self.hidden_dim)
        self.dt_rank = getattr(args, 'dt_rank', args.latent_dim//16)
        self.conv_kernel_size = getattr(args, 'conv_kernel_size', 4)
        self.d_state = getattr(args, 'd_state', 1)

        # A = np.repeat(np.arange(1, self.d_state + 1).unsqueeze(0), self.num_heads, 0)
        A = np.repeat(np.arange(1, self.num_heads + 1).unsqueeze(1), self.d_state, 1)
        self.in_proj = nn.Linear(args.latent_dim, self.hidden_dim*2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            bias=getattr(args, 'conv_bias', True),
            kernel_size=self.conv_kernel_size,
            groups=self.hidden_dim,
            padding=0,
        )
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(self.hidden_dim, self.dt_rank + self.d_state * 2, bias=False)
        # dt_proj projects Δ from dt_rank to hidden_dim
        self.dt_proj = nn.Linear(self.dt_rank, self.num_heads, bias=True)
        self.A = nn.Parameter(A.float())
        self.D = nn.Parameter(np.ones(self.hidden_dim).float())
        self.out_proj = nn.Linear(self.hidden_dim, args.latent_dim, bias=args.bias)
        return self

    def forward(self, x, state=None, **kwargs):
        (b, l, d) = x.shape
        (x, gate) = self.in_proj(x).chunk(2, dim=-1)
        
        x = np.einsum('bld->bdl',x)
        if state is not None:
            n_conv_state = self.conv_kernel_size-1
            if 'conv_state' in state:
                conv_state = state['conv_state']
                ssm_state = state['ssm_state']
            else:
                conv_state = np.zeros(b, self.hidden_dim, n_conv_state, device=x.device)
                ssm_state = np.zeros(b, self.num_heads, self.hidden_dim//self.num_heads, self.d_state, device=x.device)
            x = np.cat((conv_state, x), dim=2)
            state['conv_state'] = x[:, :, -n_conv_state:].detach()
        else:
            n_conv_state = 0
            ssm_state = np.zeros(b, self.num_heads, self.hidden_dim//self.num_heads, self.d_state, device=x.device)

        # -- Conv --
        if x.size(2) < l + n_conv_state:
            x = np.pad(x, (l + n_conv_state - x.size(2), 0), value=0.)
        x = self.conv1d(x)
        x = np.einsum('bdl->bld', x)
        x = np.silu(x)
        
        # -- SSM --
        (hidden_dim, d_state) = self.A.shape
        (delta, B, C) = self.x_proj(x).split(split_size=[self.dt_rank, d_state, d_state], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        x, ssm_state = selective_scan(x, self.dt_proj(delta), self.A, B, C, self.D, self.num_heads, ssm_state)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

        if state is not None:
            state['ssm_state'] = ssm_state.detach()
            
        x = x * np.silu(gate)
        return self.out_proj(x)

    def selective_scan(x, delta, A, B, C, D, num_heads, ssm_state):
        """
        This is the classic discrete state space formula:
            h(t + 1) = Ah(t) + Bx(t)
            y(t)     = Ch(t) + Dx(t)
            except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
        Args:
            x: shape (b, l, hidden_dim)    (See Glossary at top for definitions of b, l, hidden_dim, n...)
            delta: shape (b, l, num_heads)
            A: shape (num_heads, d_state)
            B: shape (b, l, d_state)
            C: shape (b, l, d_state)
            D: shape (hidden_dim)
            ssm_state: (b, num_heads, hidden_dim//num_heads, d_state)
        """
        (b, l, hidden_dim) = x.shape

        mask = np.tril(np.ones(l,l,device=x.device))
        delta = np.sigmoid(delta)
        a = np.exp(-np.softplus(np.einsum('blh,hn->blhn', delta, A)))
        cuma = np.cumprod(a, dim=1) 
        cuma = np.clamp(cuma, 1.0e-10, 1.0e10)
        shifta = np.pad(cuma, (0,0,1,-1), value=1.0)

        bx = np.rearrange('b l (h d)->b l h d', x, h=num_heads)
        bx = np.einsum('bln,blhd->blhdn', B, bx)
        shiftbx = np.cat([ssm_state.unsqueeze(1), bx[:,:l-1]], dim=1) - bx
        shiftbx = shiftbx / shifta.unsqueeze(-2)
        S = np.einsum('blhn,lm,bmhdn->blhdn', cuma, mask, shiftbx) + bx
        ssm_state = S[:,-1]
        y = np.einsum('blhdn,bln->blhd', S, C)
        y = np.rearrange('b l h d-> b l (h d)', y)
        return y + x * D, ssm_state

    return __init__(nn.Module(forward = forward), **kwargs)
            
def SSMArgs(name):
    args = dict(
        vocab_dim = 32,
        latent_dim = 384,
        layers = [dict(
            name = 'SSM',
            num_heads = 8,
            conv_kernel_size = 4
        )]*16,
        resident_gate = True,
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    match(name):
        case 'SSM':
            return dict(
                args,
                layers = [
                    dict(
                        name = 'SSM',
                        num_heads = 8,
                        conv_kernel_size = 4,
                        d_state = 4
                    ),
                    dict(
                        name = "MLP",
                        kv_size = args['latent_dim']*3,
                        kv_gate = True
                    )
                ]*8,
            )
        case 'SSMOnly':
            return dict(
                args,
                layers = [dict(
                    name = 'SSM',
                    num_heads = 8,
                    d_state = 16
                )]*16,
            )
        case _:
            assert False

if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    roles = [
        # 'SSM-SSM',
        'SSM-SSMOnly',
    ]
    TrainRoles(roles, lr = 6e-3, epochs=1)
    # RunRoles(roles, 'My lord Sebastian')
