from .SGNet import SGNet
from .SGNet_CVAE import SGNet_CVAE
from .SGNet_Gaussian import SGNet_Gaussian

_META_ARCHITECTURES = {
    'SGNet':SGNet,
    'SGNet_CVAE':SGNet_CVAE,
    'SGNet_Gaussian':SGNet_Gaussian
}


def build_model(args):
    meta_arch = _META_ARCHITECTURES[args.model]
    return meta_arch(args)
