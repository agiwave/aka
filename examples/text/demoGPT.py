
# import os
# os.environ["aka_provider_name"] = "aka.providers.torch"

if __name__ == "__main__":
    args = dict(
        name = 'gpt2',
        vocab_dim = 64,
        latent_dim = 768,
        dropout = 0.0,
        bias = False, # do we use bias inside LayerNorm and Linear layers?
        layers = [
            dict(
                name = 'Attention',
                window_size = 256,
                num_heads = 12,
                num_kv_groups = 12,
                rotary_embedding = True,
            ), 
            dict(
                name = 'Xproj',
                hidden_dim = 768*4
            )
        ]*6
    )

    from RomeArena import TrainRoles, RunRoles
    TrainRoles([args], lr=1.0e-3)
    RunRoles([args], "you know Caius Marcius is")
