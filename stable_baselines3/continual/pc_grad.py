import random
from typing import Iterable

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer


def pc_grad_losses(
        losses: list[Tensor],
        optimizer: Optimizer,
        params: Iterable[Parameter],
) -> None:
    grads:    list[list[Tensor]] = [[] for _ in params]
    grads_pc: list[list[Tensor]] = [[] for _ in params]

    for loss in losses:
        optimizer.zero_grad()
        loss.backward()

        for i, param in enumerate(params):
            assert param.grad is not None
            grads[i].append(param.grad.clone())
            grads_pc[i].append(param.grad.clone())

    task_ixs = list(range(len(grads)))
    random.shuffle(task_ixs)

    for ix in range(len(grads)):
        for jx  in task_ixs:
            if jx == ix: continue

            for param_ix, _ in enumerate(params):
                g_i_pc = grads_pc[param_ix][ix]
                g_j    = grads[param_ix][jx]

                cos_sim = torch.dot(g_i_pc, g_j)

                if cos_sim < 0:
                    grads_pc[param_ix][ix] -= cos_sim / torch.dot(g_j, g_j) * g_j

    for i, param in enumerate(params):
        param.grad = sum(grads[i]) # type: ignore


