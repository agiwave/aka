if __name__ == "__main__":
    from RomeArena import TrainRoles, RunRoles
    roles = [
        dict(
            name = 'tinyGPT450m',
            vocab_dim = 64,         # vs 2048
            latent_dim = 2048,
            prev_norm = 'scaledk',
            xproj_heads = 16,
            layers = [
                dict(
                    name='Hawk',
                    hidden_dim = 16384,
                    num_heads = 16,  # vs 8
                    activation = 'gelu',
                    gate = 'go',     # vs gh
                    mixers = [
                        dict(
                            name = 'Conv1d'
                        )
                    ]
                )
            ]*48,                    # vs 36
            dropout = 0.1,
            bias = False
        )
    ]
    l = TrainRoles(roles, lr=6e-3, epochs=1)
    # RunRoles(roles, 'My lord Sebastian')

    # RunRoles([
    #     'Prediction-base'
    # ], "Paul Daniels (born 4 June 1981 in Burlington)")

