"""재현성 보장을 위한 시드 고정 유틸."""
import random
from typing import Optional

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """모든 랜덤 소스에 동일한 시드를 적용한다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from transformers import set_seed
    set_seed(seed)
