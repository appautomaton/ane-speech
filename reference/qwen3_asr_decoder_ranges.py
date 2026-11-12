"""Activation ranges of the Qwen3-ASR text decoder in MLX, against fp16's limit.

The ANE computes in fp16, whose largest finite value is 65504. This transcribes
clips with mlx-speech's reference (float32) decoder path and records, per layer,
the largest magnitude each probed tensor reaches over the prefill and every
generated token. Run it from the repo root under mlx-speech's environment; MLX
needs Metal, so run it outside any sandbox:

    uv run --project ../mlx-speech python -m reference.qwen3_asr_decoder_ranges [wav ...]

RMSNorm squares its input, so a residual above 256 overflows fp16 inside the
norm unless the graph scales it down first; the table flags that separately.
The probes replace a few decoder methods in this process only; the replacements
reproduce mlx-speech's code paths, and mlx-speech's files are untouched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import mlx_speech.models.qwen3_asr.text_decoder as td
from mlx_speech.generation.qwen3_asr import Qwen3ASRTranscriber

from convert.qwen3_asr import add_checkpoint_argument, model_dir, sample_wav

FP16_MAX = 65504.0
NORM_MAX = FP16_MAX**0.5  # x^2 overflows fp16 above this

maxima: dict[str, mx.array] = {}


def probe(name: str, x: mx.array) -> mx.array:
    peak = mx.max(mx.abs(x.astype(mx.float32)))
    maxima[name] = peak if name not in maxima else mx.maximum(maxima[name], peak)
    return x


def install_probes() -> None:
    apply_attention = td.Qwen3ASRTextAttention._apply_attention

    def probed_attention(self, q, k, v, **kwargs):
        i = self.layer_idx
        probe(f"{i}.q", q), probe(f"{i}.k", k), probe(f"{i}.v", v)
        keys = td._repeat_kv(k, self.kv_repeat).astype(mx.float32)
        probe(f"{i}.logits", mx.matmul(q.astype(mx.float32), keys.transpose(0, 1, 3, 2)) * self.scale)
        return probe(f"{i}.attn_out", apply_attention(self, q, k, v, **kwargs))

    def probed_mlp(self, x, *, native=False):
        i = self.probe_index
        gate = probe(f"{i}.gate", td._linear_forward(self.gate_proj, x, native=native))
        up = probe(f"{i}.up", td._linear_forward(self.up_proj, x, native=native))
        hidden = probe(f"{i}.ffn_hidden", td._activation(gate, self.hidden_act) * up)
        return probe(f"{i}.ffn_out", td._linear_forward(self.down_proj, hidden, native=native))

    def probed_layer(method: str):
        def run(self, hidden_states, *, compute=None, **kwargs):
            compute = td._resolve_compute(compute)
            i = self.self_attn.layer_idx
            attn = getattr(self.self_attn, method)
            h = hidden_states + attn(self.input_layernorm(hidden_states), compute=compute, **kwargs)
            probe(f"{i}.resid", h)
            h = h + self.mlp(self.post_attention_layernorm(h), native=compute.native_linear)
            return probe(f"{i}.resid", h)

        return run

    def probed_lm(method):
        def run(self, *args, **kwargs):
            out = method(self, *args, **kwargs)
            probe("lm_logits", out.logits)
            mx.eval(list(maxima.values()))
            return out

        return run

    td.Qwen3ASRTextAttention._apply_attention = probed_attention
    td.Qwen3ASRTextMLP.__call__ = probed_mlp
    td.Qwen3ASRTextDecoderLayer.prefill = probed_layer("prefill")
    td.Qwen3ASRTextDecoderLayer.decode_step = probed_layer("decode_step")
    td.Qwen3ASRTextForCausalLM.prefill = probed_lm(td.Qwen3ASRTextForCausalLM.prefill)
    td.Qwen3ASRTextForCausalLM.decode_step = probed_lm(td.Qwen3ASRTextForCausalLM.decode_step)


def fmt(v: float) -> str:
    return f"{v:,.1f}"


def flag(v: float, limit: float = FP16_MAX) -> str:
    return " **over**" if v > limit else " *near*" if v >= limit / 10 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wavs", type=Path, nargs="*", default=[sample_wav()])
    add_checkpoint_argument(parser)
    args = parser.parse_args()

    install_probes()
    asr = Qwen3ASRTranscriber.from_dir(model_dir(args.checkpoint))
    for i, layer in enumerate(asr.model.text_decoder.model.layers):
        layer.mlp.probe_index = i
    for wav in args.wavs:
        result = asr.transcribe(wav)
        print(f"{wav.name}: {result.prompt_tokens} prompt tokens, {len(result.tokens)} generated: {result.text}")

    values = {k: v.item() for k, v in maxima.items()}
    layers = len(asr.model.text_decoder.model.layers)
    print("\n| Layer | Residual | Attn logits | MLP gate·up | Largest other |")
    print("| ---: | ---: | ---: | ---: | --- |")
    over_norm = []
    for i in range(layers):
        get = lambda n: values[f"{i}.{n}"]  # noqa: E731
        other = max(("q", "k", "v", "attn_out", "gate", "up", "ffn_out"), key=get)
        resid = get("resid")
        if resid > NORM_MAX:
            over_norm.append(i)
        print(f"| {i} | {fmt(resid)}{flag(resid)} | {fmt(get('logits'))}{flag(get('logits'))} "
              f"| {fmt(get('ffn_hidden'))}{flag(get('ffn_hidden'))} | {other} {fmt(get(other))}{flag(get(other))} |")
    print(f"\nvocab logits: {fmt(values['lm_logits'])}{flag(values['lm_logits'])}")
    peak = max(values, key=values.get)
    print(f"peak: {peak} {fmt(values[peak])} ({FP16_MAX / values[peak]:.1f}x below fp16 max)")
    print(f"residual above {NORM_MAX:.0f} (RMSNorm squares overflow fp16 unless scaled): "
          f"layers {over_norm or 'none'}")


if __name__ == "__main__":
    main()
