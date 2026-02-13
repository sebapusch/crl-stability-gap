import numpy as np
from stable_baselines3.common.logger import KVWriter
import wandb
from typing import Any

class WandbWriter(KVWriter):
    def write(self, key_values: dict[str, Any], key_excluded: dict[str, Any], step: int = 0) -> None:
        log_dict: dict[str, Any] = {}
        for k, v in key_values.items():
            if v is None:
                continue
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            if isinstance(v, (int, float)):
                log_dict[k] = v

        if log_dict:
            wandb.log(log_dict, step=step)


