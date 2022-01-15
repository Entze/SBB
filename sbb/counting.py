import operator
from functools import reduce
from typing import Union

import torch
from torch import Tensor


def count_elems(t: Union[Tensor, torch.Size]) -> int:
    if isinstance(t, Tensor):
        t = t.size()
    return reduce(operator.mul, t, 1)


def enumerate_sequences(*args,
                        shift: int = 8) -> Tensor:  # IDEA: make shift broadcastable (i.e. shift according to space requirement)
    if not args:
        raise ValueError("At least one tensor required")
    worklist = [arg for arg in args]
    with torch.no_grad():
        t = worklist.pop()
        if isinstance(t, tuple):
            replicate, tensor = t
            for i in range(replicate - 1):
                worklist.append(torch.clone(tensor))
            t = tensor
        result = torch.clone(t)
        while worklist:
            t = worklist.pop()
            if isinstance(t, tuple):
                replicate, tensor = t
                for i in range(replicate):
                    worklist.append(torch.clone(tensor))
                continue
            for _d in result.size():
                t = torch.unsqueeze(t, -1)
            result = (result << shift) + t

    return result


def count_sequences(*args) -> int:
    if not args:
        return 0
    result = 1
    index = 0
    while index < len(args):
        t = args[index]
        if isinstance(t, tuple):
            r, tensor = t
            t = tensor
        else:
            r = 1
        result *= count_elems(t) ** r
        index += 1
    return result


def enumerate_arrangements(*args, shift: int = 8) -> Tensor:
    pass