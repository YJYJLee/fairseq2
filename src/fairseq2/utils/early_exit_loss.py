# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import numpy as np
import torch
from enum import Enum
from typing import Dict, List, Optional
from torch import Tensor

from fairseq2.nn.padding import PaddingMask

hidden_states_dict: Dict[int, Tensor] = {}

def hook(
    layer_idx: int,
    layer_output: Tensor,
    layer_padding_mask: Optional[PaddingMask],
    num_layers: int,
) -> bool:
    global hidden_states_dict

    # TODO: handle only doing early exit loss on specific layers
    hidden_states_dict[layer_idx] = layer_output

    return True


class LossScaleType(str, Enum):
    ONE = "one"
    L = "l"
    L2 = "l2"
    SUM_L = "sum_l"
    SUM_L2 = "sum_l2"
    INV_L = "inv_l"
    SQRT_L = "sqrt_l"
    INV_SQRT_L = "inv_sqrt_l"

def early_exit_loss(model, hidden_states_dict, batch, loss_fn, e_scale: float=20.0, loss_scale_type=LossScaleType.ONE):
    hidden_states = tuple(hidden_states_dict.values())
    hidden_layer_ids = tuple(hidden_states_dict.keys())

    losses_early = []
    for hidden_state in hidden_states:
        logits_early = model.module.model.final_proj(model.module.model.text_decoder.layer_norm(hidden_state))
        # FIXME: assuming that the other output is None, which is not always the case
        loss_early = loss_fn(batch, *(logits_early, None))
        losses_early.append(loss_early)  

    losses_early = torch.stack(losses_early, dim=0)
    losses_scales = layer_ids_to_loss_scales(torch.Tensor(hidden_layer_ids).to(losses_early), len(model.module.model.text_decoder.layers), loss_scale_type, e_scale)

    return torch.sum(losses_scales * losses_early)

def layer_ids_to_loss_scales(layer_ids, n_layers, loss_scale_type: LossScaleType, e_scale: float):
    match loss_scale_type:
        case LossScaleType.ONE:
            loss_scales = torch.ones(len(layer_ids))
        case LossScaleType.L:
            loss_scales = torch.Tensor(layer_ids+1)
        case LossScaleType.L2:
            loss_scales = torch.Tensor((layer_ids+1)**2)
        case LossScaleType.SUM_L:
            # TODO: should we change to sum 0:i ? Perhaps create a new scale_type
            loss_scales = torch.cumsum(layer_ids+1, dim=0)
        case LossScaleType.SUM_L2:
            # TODO: should we change to sum 0:i ? Perhaps create a new scale_type
            loss_scales = torch.cumsum((layer_ids+1)**2, dim=0)
        case LossScaleType.SQRT_L:
            loss_scales = torch.sqrt(layer_ids+1)
        case LossScaleType.INV_L:
            loss_scales = 1.0 / (layer_ids+1)
        case LossScaleType.INV_SQRT_L:
            loss_scales = 1.0 / torch.sqrt(layer_ids+1)
        case _:
            raise ValueError(f"Unsupported loss_scale type {loss_scale_type}")

    loss_scales = loss_scales * torch.where(layer_ids < n_layers - 1, e_scale, 1.0)
    # normalize loss scales to ensure that their sum is 1.0
    loss_scales = loss_scales / torch.sum(loss_scales)
    assert torch.isclose(torch.sum(loss_scales), torch.Tensor([1.0]).to(loss_scales))

    return loss_scales

class EarlyExitCurriculumType(str, Enum):
    NONE = "none"
    ROTATIONAL = "rot"
    GRADUAL = "gradual"

def build_early_exit_curriculum(early_exit_curriculum: EarlyExitCurriculumType, *args, **kwargs):
    match early_exit_curriculum:
        case EarlyExitCurriculumType.NONE:
            return None

        case EarlyExitCurriculumType.ROTATIONAL:
            return RotationalEarlyExitCurriculum(*args, **kwargs)

        case EarlyExitCurriculumType.GRADUAL:
            return GradualEarlyExitCurriculum(*args, **kwargs)

        case _:
            raise ValueError(f"Unsupported early loss curriculum {early_exit_curriculum}.")
    

# TODO: create a base curriculum class that can be used for other aspects, e.g., dropout, datasets, etc.
class EarlyExitCurriculum():
    def __init__(self, output_hidden_states, max_steps, verbose=False):
        self._init_output_hidden_states = output_hidden_states
        self.output_hidden_states = output_hidden_states
        self.verbose = verbose
        self.max_steps = max_steps

    def step(self):
        pass

    def get(self):
        return self.output_hidden_states

class RotationalEarlyExitCurriculum(EarlyExitCurriculum):
    def __init__(self, output_hidden_states, max_steps, verbose=False):
        super().__init__(output_hidden_states, max_steps, verbose)

    def step(self):
        self.output_hidden_states = np.roll(self.output_hidden_states, -1)
        if self.verbose:
            print(f"Updating self.output_hidden_states to {self.output_hidden_states}.")

class GradualEarlyExitCurriculum(EarlyExitCurriculum):
    def __init__(self, output_hidden_states, max_steps, verbose=False):
        super().__init__(output_hidden_states, max_steps, verbose)
        self._step = 0

    def step(self):
        percent_trained = self._step / self.max_steps
        n_layers = len(self.output_hidden_states)
        for layer_index in range(len(self.output_hidden_states)):
            # TODO: replace 2 with an argument
            should_train = (percent_trained * 2) >= ((n_layers - 1 - layer_index) / (n_layers - 1))
            self.output_hidden_states[layer_index] = should_train

        # TODO: move this to step() in parent class?
        # TODO: how to ensure we always call parent step() in derived class?
        self._step += 1
        if self.verbose:
            print(f"Updating self.output_hidden_states to {self.output_hidden_states}.")
