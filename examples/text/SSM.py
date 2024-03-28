
import aka.nn as nn
import aka.numpy as np

def SSMBlock(**kwargs):
    '''

    '''
    def __init__(self, **kwargs):
        args = nn.Object(**kwargs)
        bias = getattr(args, 'bias', False)

        self.hidden_dim = getattr(args, 'hidden_dim', args.latent_dim)
        self.num_state = getattr(args, 'num_state', 16)
        self.state_dim = getattr(args, 'state_dim', self.hidden_dim//self.num_state)

        self.in_proj = nn.Linear(args.latent_dim, self.hidden_dim*2, bias=args.bias)
        self.conv_kernel_size = getattr(args, 'conv_kernel_size', 4)
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            bias=args.bias,
            kernel_size=self.conv_kernel_size,
            groups=self.hidden_dim, # ？？？？？？？？？？？？
            padding=0,
        )
        self.out_proj = nn.Linear(self.hidden_dim, args.latent_dim, bias=bias)
        self.dropout = nn.Dropout(getattr(args, 'dropout', 0.2))

        self.A = nn.Parameter(np.arange(1, 1+self.num_state, dtype=np.float))
        self.B = nn.Parameter(np.zeros(self.num_state, self.state_dim))
        self.CN = nn.Parameter(shape=(self.num_state, self.hidden_dim), initializer='xavier_uniform')
        self.CD = nn.Parameter(shape=(self.state_dim, self.hidden_dim), initializer='xavier_uniform')
        self.D = nn.Parameter(np.zeros(self.hidden_dim))
        self.ON = nn.Parameter(shape=(self.hidden_dim, self.num_state), initializer='xavier_uniform')
        self.OD = nn.Parameter(shape=(self.hidden_dim, self.state_dim), initializer='xavier_uniform')
        return self

    def forward(self, inputs, *, state=None, **kwargs):
        x = inputs
        (b, l, d) = x.shape
        (x, gate) = self.in_proj(x).chunk(2, dim=-1)
        
        convx = np.rearrange('b l d->b d l',x)
        if state is not None:
            n_conv_state = self.conv_kernel_size-1
            if 'conv_state' in state:
                ssm_state = state['ssm_state']
                convx = np.cat((state['conv_state'], convx), dim=2)
            else:
                ssm_state = np.zeros(b, 1, self.num_state, self.state_dim, device=x.device)
        else:
            n_conv_state = 0
            ssm_state = np.zeros(b, 1, self.num_state, self.state_dim, device=x.device)

        if convx.size(2) < l + n_conv_state:
            convx = np.pad(convx, (l + n_conv_state - convx.size(2), 0), value=0.)
        x = self.conv1d(convx)
        x = np.silu(x)
        x = np.rearrange('b d l->b l d',x)

        O = np.einsum('vh,vd->vhd', self.ON,self.OD)
        C = np.einsum('hv,dv->hdv', self.CN,self.CD)
        logA = -np.softplus(self.A + np.einsum('vhd,hdv->h', O, C))
        mask = np.tril(np.ones(l,l,device=x.device))  # [L,L] 
        logA = np.einsum('h,lm->hlm',logA,mask)
        cumA = np.exp(np.cumsum(logA, dim=1))

        sumB = self.B + np.einsum('vhd,blv->blhd', O, x-self.D)
        shiftB = np.cat([ssm_state, sumB[:,:l-1]], dim=1)

        y = np.einsum('hlm,bmhd,hdv->blv',cumA, shiftB, C)
        y += np.einsum('blhd,hdv->blv', sumB, C) + self.D
        if state is not None:
            state['conv_state'] = convx[:, :, -n_conv_state:].detach()
            state['ssm_state'] = (np.einsum('hlm,bmhd->blhd',cumA[:,-1:], shiftB) + sumB[:,-1:]).detach()

        y = y * np.gelu(gate)
        return self.dropout(self.out_proj(y))
    return __init__(nn.Module(forward=forward), **kwargs)

def SSMArgs(name):
    return dict(
        vocab_dim = 32,
        latent_dim = 384,
        layers = [
            dict(
                name = 'SSM',
                windows_size = 64,  # Limit Attention Seq Length to 256. Gemma2b --> 8192
                qk_dim = 384,
                num_heads = 8,
                num_kv_groups = 8,
                rotary_embedding = True,
                num_state = 64
            ), 
            dict(
                name = 'MLP',
                kv_size = 384 * 3,
                kv_gate = False,
            )
        ]*8,
        dropout = 0.1,
        bias = False, # bias in Linear?
    )
    return args

if __name__ == "__main__":
    from RomeArena import TrainArena
    TrainArena([
        # 'Gemma-20m', 
        'SSM-Base',
    ], lr = 6e-4, epochs=4)
