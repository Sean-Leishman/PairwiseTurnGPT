import torch
from data import TurnType, TurnEndType


def parse_kwargs(labels, **kwargs):
    for key, value in kwargs.items():
        if value is None:
            kwargs[key] = torch.zeros_like(labels)
        else:
            kwargs[key] = value
    return kwargs


def get_mask(
    config,
    labels,
    device,
    yield_mask=None,
    non_yield_mask=None,
    ignore_mask=None,
    overlap_mask=None,
    bc_mask=None,
    interrupt_mask=None,
    interrupt_mask_overlap=None,
    interrupt_mask_bc=None,
    interrupt_mask_yield=None,
    **kwargs,
):
    yield_mask = yield_mask if yield_mask is not None else torch.zeros_like(labels)
    non_yield_mask = (
        non_yield_mask if non_yield_mask is not None else torch.zeros_like(labels)
    )
    ignore_mask = ignore_mask if ignore_mask is not None else torch.ones_like(labels)
    overlap_mask = (
        overlap_mask if overlap_mask is not None else torch.zeros_like(labels)
    )
    bc_mask = bc_mask if bc_mask is not None else torch.zeros_like(labels)

    if config == TurnEndType.YIELD:
        mask = torch.logical_and(ignore_mask, yield_mask)
    elif config == TurnEndType.NORMAL:
        mask = torch.logical_and(ignore_mask, non_yield_mask)
    elif config == TurnType.BACKCHANNEL:
        mask = torch.logical_and(ignore_mask, bc_mask)
    elif config == TurnType.OVERLAP:
        mask = torch.logical_and(ignore_mask, overlap_mask)
    elif config == TurnType.NON_OVERLAP:
        mask = torch.logical_and(ignore_mask, torch.logical_not(overlap_mask))
    else:
        mask = torch.logical_and(ignore_mask, ignore_mask).to(device)

    return mask[..., 1:]


def get_interrupt_mask(
    config,
    labels,
    interrupt_mask=None,
    interrupt_mask_normal=None,
    interrupt_mask_overlap=None,
    interrupt_mask_bc=None,
    interrupt_mask_yield=None,
    **kwargs,
):
    interrupt_mask = (
        interrupt_mask if interrupt_mask is not None else torch.zeros_like(labels)
    )
    interrupt_mask_normal = (
        interrupt_mask_normal
        if interrupt_mask_normal is not None
        else torch.zeros_like(labels)
    )
    interrupt_mask_overlap = (
        interrupt_mask_overlap
        if interrupt_mask_overlap is not None
        else torch.zeros_like(labels)
    )
    interrupt_mask_bc = (
        interrupt_mask_bc if interrupt_mask_bc is not None else torch.zeros_like(labels)
    )
    interrupt_mask_yield = (
        interrupt_mask_yield
        if interrupt_mask_yield is not None
        else torch.zeros_like(labels)
    )

    interrupt_mask = interrupt_mask[..., 1:]
    if config == TurnType.BACKCHANNEL:
        interrupt_mask = interrupt_mask_bc[..., 1:]
    elif config == TurnType.OVERLAP:
        interrupt_mask = interrupt_mask_overlap[..., 1:]
    elif config == TurnEndType.YIELD:
        interrupt_mask = interrupt_mask_yield[..., 1:]
    elif config == TurnEndType.NORMAL:
        interrupt_mask = interrupt_mask_normal[..., 1:]

    return interrupt_mask
