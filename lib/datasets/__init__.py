# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .aflw import AFLW
from .cofw import COFW
from .cofwsd import COFWSD
from .face300w import Face300W
from .face300wsd import Face300WSD
from .wflw import WFLW
from .wflwsd import WFLWSD
from .wflwe70 import WFLWE70
from .free import FreeData

__all__ = ['AFLW', 'COFW', 'Face300W', 'WFLW', 'get_dataset']


def get_dataset(config):

    if config.DATASET.DATASET == 'AFLW':
        return AFLW
    elif config.DATASET.DATASET == 'COFW':
        return COFW
    elif config.DATASET.DATASET == 'COFWSD':
        return COFWSD
    elif config.DATASET.DATASET == '300W':
        return Face300W
    elif config.DATASET.DATASET == '300WSD':
        return Face300WSD
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW
    elif config.DATASET.DATASET == 'WFLWSD':
        return WFLWSD
    elif config.DATASET.DATASET == 'WFLWE70':
        return WFLWE70
    elif config.DATASET.DATASET == 'FreeData':
        return FreeData
    else:
        raise NotImplemented()

