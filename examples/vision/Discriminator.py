import aka.nn as nn

def Discriminator(image_channels, num_filters_last=64, n_layers=3):
    '''
    From VQGAN
    '''
    layers = [nn.Conv2d(image_channels, num_filters_last, 4, 2, 1), nn.LeakyReLU(0.2)]
    num_filters_mult = 1

    for i in range(1, n_layers + 1):
        num_filters_mult_last = num_filters_mult
        num_filters_mult = min(2 ** i, 8)
        layers += [
            nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 4,
                        2 if i < n_layers else 1, 1, bias=False),
            nn.BatchNorm2d(num_filters_last * num_filters_mult),
            nn.LeakyReLU(0.2, True)
        ]

    layers.append(nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1))
    return nn.Sequential(*layers)
