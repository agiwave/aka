import aka.nn as nn
import aka.numpy as np

def MambaArgs(name):
    args = nn.Args(
        latent_dim = 384
    )
    mamba_args = nn.Args(
        name = 'Mamba',
        conv_kernel_size = 4,
        conv_bias = True
    )
    match name:
        case '20m':
            args.latent_dim = 384
            mamba_args.qk_dim = 384
            mamba_args.d_state = 16
            args.layers = [mamba_args]*20
        case _:
            assert False
    mamba_args.dt_rank = args.latent_dim // 16
    return args

def MambaBlock(args):
    """A single Mamba block, as described in Figure 3 in Section 3.4 in the Mamba paper [1]."""
    def __init__(self, args):
        mamba_args = nn.Args(
            qk_dim = args.qk_dim,
            dt_rank = args.dt_rank,
            conv_kernel_size = args.conv_kernel_size,
            conv_bias = args.conv_bias,
            d_state = args.d_state
        )
        A = np.repeat(np.arange(1, mamba_args.d_state + 1).unsqueeze(0), mamba_args.qk_dim, 0)
        self.args = mamba_args
        self.in_proj = nn.Linear(args.latent_dim, mamba_args.qk_dim*2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=mamba_args.qk_dim,
            out_channels=mamba_args.qk_dim,
            bias=mamba_args.conv_bias,
            kernel_size=mamba_args.conv_kernel_size,
            groups=mamba_args.qk_dim,
            padding=mamba_args.conv_kernel_size-1,
        )
        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = nn.Linear(mamba_args.qk_dim, mamba_args.dt_rank + mamba_args.d_state * 2, bias=False)
        # dt_proj projects Δ from dt_rank to qk_dim
        self.dt_proj = nn.Linear(mamba_args.dt_rank, mamba_args.qk_dim, bias=True)
        self.A_log = nn.Parameter(np.log(A))
        self.D = nn.Parameter(np.ones(mamba_args.qk_dim))
        self.out_proj = nn.Linear(mamba_args.qk_dim, args.latent_dim, bias=args.bias)
        return self

    def forward(self, x, state=None, **kwargs):
        (b, l, d) = x.shape
        (x, gate) = self.in_proj(x).chunk(2, dim=-1)
        
        x = np.einsum('bld->bdl',x)
        if state is not None:
            n_conv_state = self.args.conv_kernel_size-1
            if 'conv_state' in state:
                conv_state = state['conv_state']
                ssm_state = state['ssm_state']
            else:
                conv_state = np.zeros(b, self.args.qk_dim, n_conv_state, device=x.device)
                ssm_state = np.zeros(b, self.args.qk_dim, self.args.d_state, device=x.device)
            x = np.cat((conv_state, x), dim=2)
            y = self.conv1d(x)[:, :, n_conv_state:n_conv_state+l]
            y = np.einsum('bdl->bld', y)
            y = np.silu(y)
            y, ssm_state = self.ssm(y, ssm_state)
            state['ssm_state'] = ssm_state.detach()
            state['conv_state'] = x[:, :, -n_conv_state:].detach()
        else:
            n_conv_state = 0
            ssm_state = np.zeros(b, self.args.qk_dim, self.args.d_state, device=x.device)
            y = self.conv1d(x)[:, :, n_conv_state:n_conv_state+l]
            y = np.einsum('bdl->bld', y)
            y = np.silu(y)
            y, ssm_state = self.ssm(y,ssm_state)

        y = y * np.silu(gate)
        return self.out_proj(y)

    def ssm(self, x, ssm_state):
        """
        Args:
            x: (b, l, qk_dim)
        Returns:
            output: shape (b, l, qk_dim)
        """
        (qk_dim, d_state) = self.A_log.shape
        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        A = -np.exp(self.A_log.float())  # shape (qk_dim, n)
        D = self.D.float()
        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*d_state)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, d_state, d_state], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = np.softplus(self.dt_proj(delta))  # (b, l, qk_dim)
        return selective_scan(x, delta, A, B, C, D, ssm_state)  # This is similar to run_SSM(A, B, C, u) in The Annotated S4 [2]

    def selective_scan(x, delta, A, B, C, D, ssm_state):
        """
        This is the classic discrete state space formula:
            h(t + 1) = Ah(t) + Bx(t)
            y(t)     = Ch(t) + Dx(t)
            except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
        Args:
            x: shape (b, l, qk_dim)    (See Glossary at top for definitions of b, l, qk_dim, n...)
            delta: shape (b, l, qk_dim)
            A: shape (qk_dim, d_state)
            B: shape (b, l, d_state)
            C: shape (b, l, d_state)
            D: shape (qk_dim)
            ssm_state: (b, qk_dim, d_state)
        Returns:
            output: shape (b, l, qk_dim), ssm_state
        """
        (b, l, qk_dim) = x.shape
        
        deltaA = np.exp(np.einsum('bld,dn->bldn', delta, A))
        deltaBX = np.einsum('bld,bln,bld->bldn', delta, B, x)

        # -- TODO, how to fast it in parallel? --
        y_list = []
        for i in range(l):
            ssm_state = deltaA[:, i] * ssm_state + deltaBX[:, i]
            y = np.einsum('bdn,bn->bd', ssm_state, C[:, i])
            y_list.append(y)
        y = np.stack(y_list, dim=1)
        return y + x * D, ssm_state

    return __init__(nn.Module(forward = forward, ssm = ssm, selective_scan = selective_scan),args)
            

def Mamba(name):
    import aka.repo as repo
                
    # -- Tokenizer --
    tokenizer = repo.AutoTokenizer(name)
    cfg = repo.fopen(name, 'config.json', ftype='json')
    args = nn.Args(
        tokenizer = tokenizer,
        vocab_size = cfg['vocab_size'],
        latent_dim = cfg['d_model'],
        layers = [
            nn.Args(
                name = 'Mamba',
                qk_dim = cfg['intermediate_size'],
                dt_rank = cfg['d_model']//16,
                conv_kernel_size = 4,
                conv_bias = True,
                d_state = cfg['state_size']
            )
        ]*cfg['n_layer'],
        bias = False
    )

    # -- Model --
    from CausalLM import CausalLM
    mamba = CausalLM(args)
    if repo.exist(name, "model.safetensors"):
        with repo.fopen(name, "model.safetensors", ftype='safetensor') as f:
            with np.no_grad():
                mamba.embedding.weight.copy_(f.get_tensor('backbone.embeddings.weight'))
                mamba.post_norm.weight.copy_(f.get_tensor('backbone.norm_f.weight'))
                for i in range(len(mamba.layers)):
                    mamba.layers[i].norm.weight.copy_(f.get_tensor(f'backbone.layers.{i}.norm.weight'))
                    mamba.layers[i].layer.A_log.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.A_log'))
                    mamba.layers[i].layer.D.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.D'))
                    mamba.layers[i].layer.conv1d.weight.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.conv1d.weight'))
                    mamba.layers[i].layer.conv1d.bias.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.conv1d.bias'))
                    mamba.layers[i].layer.dt_proj.weight.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.dt_proj.weight'))
                    mamba.layers[i].layer.dt_proj.bias.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.dt_proj.bias'))
                    mamba.layers[i].layer.in_proj.weight.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.in_proj.weight'))
                    mamba.layers[i].layer.out_proj.weight.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.out_proj.weight'))
                    mamba.layers[i].layer.x_proj.weight.copy_(f.get_tensor(f'backbone.layers.{i}.mixer.x_proj.weight'))
    return mamba

if __name__ == "__main__":
    mamba = Mamba('data/mamba-370m-hf')
    print('Model loaded')
    for w in mamba.generator("Mamba is"):
        print(w, end='')


