"""Replay a traced MLX graph as Core AI ops, with every float in fp16.

`reference/` traces mlx-speech's modules with `mx.export_function`. This emits
each traced primitive as `coreai.authoring` ops. Two rewrites keep the graph on
the ANE; without them the ANE compiler rejects it and Core AI silently runs the
whole graph on the GPU:

- The ANE compiler rejects softmax output as the left operand of a dynamic
  matmul, so attention's P.V is emitted as (VT.PT)T.
- A graph has a fixed size, so an optional additive key mask is added before
  every softmax; a block with fewer valid tokens then attends only to those.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import coreai.authoring as ca
import numpy as np
from coreai.authoring import Value

F16, U32 = np.float16, np.uint32  # Core AI takes shapes and permutations as uint32
SHAPE_ONLY = {"Reshape", "ExpandDims", "Squeeze", "Flatten", "Unflatten"}


def bf16_to_f16(bits: np.ndarray) -> np.ndarray:
    """bf16 stored as raw uint16 bits, widened exactly to f32, then rounded to fp16."""
    return (bits.astype(np.uint32) << 16).view(np.float32).astype(F16)


@dataclass(frozen=True)
class Trace:
    """A traced function: its input and output ids and shapes, primitives, constants."""

    inputs: list[tuple[str, tuple[int, ...]]]
    outputs: list[tuple[str, tuple[int, ...]]]
    primitives: list[dict]
    constants: dict[str, np.ndarray]
    meta: dict[str, Any]

    @classmethod
    def load(cls, stem: Path) -> Trace:
        """Read `<stem>.json` and its constants, `<stem>.npz` or the file of the
        trace named in `weights`, as written by a reference/ trace script."""
        doc = json.loads(stem.with_suffix(".json").read_text())
        raw = np.load(stem.with_name(doc.pop("weights", stem.name) + ".npz"))
        constants = {}
        for name, dtype in doc.pop("constants").items():
            value = bf16_to_f16(raw[name]) if dtype == "bfloat16" else raw[name]
            if value.dtype.kind == "f":
                value = value.astype(F16)
                if not np.isfinite(value).all():
                    raise ValueError(f"constant {name} overflows fp16")
            constants[name] = value
        graph = doc.pop("graph")
        io = {r["type"]: [(i[0], tuple(i[1])) for i in r[r["type"]]] for r in graph if r["type"] in ("inputs", "outputs")}
        primitives = [r for r in graph if r["type"] == "primitive"]
        return cls(io["inputs"], io["outputs"], primitives, constants, doc)


class Translator:
    """Emits a trace's primitives in order; call it inside a `coreai.authoring` graph."""

    def __init__(self, trace: Trace, key_mask: Value | None = None) -> None:
        self.trace = trace
        self.key_mask = key_mask
        self.env: dict[str, Any] = dict(trace.constants)
        self.softmax_outputs: set[str] = set()

    def __call__(self, *inputs: Value) -> list[Value]:
        for (name, _), value in zip(self.trace.inputs, inputs, strict=True):
            self.env[name] = value
        for record in self.trace.primitives:
            if len(record["outputs"]) != 1:
                raise NotImplementedError(f"{record['name']} with several outputs")
            self.env[record["outputs"][0][0]] = self.emit(record)
        return [self.env[name] for name, _ in self.trace.outputs]

    def emit(self, record: dict) -> Any:
        name, args = record["name"], record["arguments"]
        ids = [i[0] for i in record["inputs"]]
        ins = [self.env[i] for i in ids]
        out_shape = record["outputs"][0][1]
        rank = len(out_shape)
        const = all(isinstance(i, np.ndarray) for i in ins)

        if name == "Softmax" or (name == "AsType" and ids[0] in self.softmax_outputs):
            self.softmax_outputs.add(record["outputs"][0][0])

        match name:
            case _ if name in SHAPE_ONLY:
                if const:
                    return ins[0].reshape(out_shape)
                return ca.reshape(ins[0], np.array(out_shape, U32))
            case "Transpose":
                if const:
                    return np.ascontiguousarray(ins[0].transpose(args[0]))
                return ca.transpose(ins[0], np.array(args[0], U32))
            case "Broadcast":
                if const:
                    return ins[0]  # every consumer is a broadcasting op
                return ca.broadcast_to(ins[0], np.array(args[0], U32))
            case "Slice":
                start, stop, step = args
                if const:
                    return np.ascontiguousarray(ins[0][tuple(map(slice, start, stop, step))])
                return ca.slice_(ins[0], list(start), list(stop), list(step))  # lists become index constants
            case "Concatenate":
                if const:
                    return np.concatenate(ins, axis=args[0])
                return ca.concat(args[0], list(ins))
            case "AsType":
                if "float" not in args[0]:
                    raise NotImplementedError(f"AsType to {args[0]}")
                return ins[0]  # every float is fp16
            case "LayerNorm":
                return layer_norm(*ins, args[0])
            case "RMSNorm":
                return rms_norm(*ins, args[0])
            case "Gather" if isinstance(ins[1], np.ndarray) and ins[1].ndim == 0:
                (axis,), sizes = args
                start = [0] * len(sizes)
                start[axis] = int(ins[1])
                stop = [b + n for b, n in zip(start, sizes, strict=True)]
                if const:
                    return np.ascontiguousarray(ins[0][tuple(map(slice, start, stop))])
                return ca.slice_(ins[0], start, stop, [1] * len(sizes))
            case "Negative":
                return ca.broadcasting_mul(ins[0], F16(-1))
            case "Sigmoid":
                return ca.sigmoid(ins[0])
            case "AddMM":
                if args != [1.0, 1.0]:
                    raise NotImplementedError(f"AddMM with alpha, beta = {args}")
                return ca.broadcasting_add(ca.batch_matmul(ins[0], ins[1]), ins[2])
            case "Matmul" if ids[0] in self.softmax_outputs and not const:
                p, v = ins
                return swap_last(ca.batch_matmul(swap_last(v, rank), swap_last(p, rank)), rank)
            case "Matmul":
                return ca.batch_matmul(*ins)
            case "Add":
                return ca.broadcasting_add(*ins)
            case "Multiply":
                return ca.broadcasting_mul(*ins)
            case "Divide":
                return ca.broadcasting_divide(*ins)
            case "Erf":
                return ca.erf(ins[0])
            case "Softmax":
                scores = ins[0] if self.key_mask is None else ca.broadcasting_add(ins[0], self.key_mask)
                return ca.softmax(scores, rank - 1)
            case "Convolution":
                return convolution(*ins, args)
        raise NotImplementedError(name)


def swap_last(x: Value, rank: int) -> Value:
    return ca.transpose(x, np.array([*range(rank - 2), rank - 1, rank - 2], U32))


def layer_norm(x: Value, weight: Any, bias: Any, eps: float) -> Value:
    """LayerNorm over the last axis from primitives, in fp16; ANE-safe for this model's ranges."""
    shape = x.type.shape
    axis = [len(shape) - 1]
    keep = np.array([*shape[:-1], 1], U32)
    mean = ca.reshape(ca.reduce_mean(x, axis), keep)
    centered = ca.broadcasting_sub(x, mean)
    var = ca.reshape(ca.reduce_mean(ca.mul(centered, centered), axis), keep)
    normed = ca.broadcasting_mul(centered, ca.rsqrt(ca.broadcasting_add(var, F16(eps))))
    return ca.broadcasting_add(ca.broadcasting_mul(normed, weight), bias)


def rms_norm(x: Value, weight: Any, eps: float) -> Value:
    """RMSNorm over the last axis in fp16, safe for large residuals.

    Qwen3's residual stream reaches about 1e4, whose square overflows fp16. Each
    row is first divided by its largest magnitude (floored so an all-zero padding
    row stays finite), which RMSNorm is invariant to once eps is scaled to match:
    x / rms(x) = s / sqrt(mean(s^2) + eps / m^2), with s = x / m.
    """
    shape = x.type.shape
    axis = [len(shape) - 1]
    keep = np.array([*shape[:-1], 1], U32)
    peak = ca.reshape(ca.reduce_max(ca.abs_(x), axis), keep)
    peak = ca.broadcasting_maximum(peak, F16(2**-10))
    scaled = ca.broadcasting_divide(x, peak)
    mean_square = ca.reshape(ca.reduce_mean(ca.mul(scaled, scaled), axis), keep)
    eps_root = ca.broadcasting_divide(F16(eps**0.5), peak)  # squared below; never overflows
    inv = ca.rsqrt(ca.add(mean_square, ca.mul(eps_root, eps_root)))
    return ca.broadcasting_mul(ca.broadcasting_mul(scaled, inv), weight)


def convolution(x: Value, weight: np.ndarray, args: list) -> Value:
    """MLX's NHWC convolution as Core AI's NCHW conv2d with explicit zero padding."""
    stride, pad_lo, pad_hi, kernel_dilation, input_dilation, groups, flip = args
    if any(d != 1 for d in input_dilation) or flip or not isinstance(weight, np.ndarray):
        raise NotImplementedError(f"Convolution with {args}")
    padding = np.array([0, 0, 0, 0, pad_lo[0], pad_hi[0], pad_lo[1], pad_hi[1]], U32)
    nchw = ca.pad(ca.transpose(x, np.array([0, 3, 1, 2], U32)), padding, F16(0))
    y = ca.conv2d(nchw, np.ascontiguousarray(weight.transpose(0, 3, 1, 2)), stride, kernel_dilation, groups)
    return ca.transpose(y, np.array([0, 2, 3, 1], U32))
