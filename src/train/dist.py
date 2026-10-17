import torch
import torch.distributed as dist


class _FullGather(torch.autograd.Function):
    """All-gather with gradient support using a single contiguous output buffer."""

    @staticmethod
    def forward(ctx, x):
        if not (dist.is_available() and dist.is_initialized()):
            return (x,)
        world = dist.get_world_size()
        # all_gather_into_tensor concatenates inputs along dim 0. Its output must
        # therefore be flat on that dimension, not prefixed by a world axis.
        out = x.new_empty((world * x.shape[0], *x.shape[1:]))
        dist.all_gather_into_tensor(out, x)
        return tuple(out.chunk(world, dim=0))

    @staticmethod
    def backward(ctx, *grads):
        if not (dist.is_available() and dist.is_initialized()):
            return grads[0]
        g = torch.stack(grads)
        dist.all_reduce(g)
        rank = dist.get_rank()
        return g[rank]


def all_gather_with_grad(x: torch.Tensor) -> torch.Tensor:
    """Global concat on dim 0. Autograd-enabled. No-op if not distributed."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    parts = _FullGather.apply(x)
    return torch.cat(parts, dim=0)


def all_gather_no_grad(x: torch.Tensor) -> torch.Tensor:
    """Global concat on dim 0 without grad. Local rank slice keeps grad."""
    if not (dist.is_available() and dist.is_initialized()):
        return x
    world = dist.get_world_size()
    rank = dist.get_rank()
    with torch.no_grad():
        out = x.new_empty((world * x.shape[0], *x.shape[1:]))
        dist.all_gather_into_tensor(out, x)
        parts = list(out.chunk(world, dim=0))
        parts[rank] = x  # keep local slice with grad semantics
        return torch.cat(parts, dim=0)
