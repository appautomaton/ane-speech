# Performance

A one-time record of how fast the Qwen3-ASR runtime (Confucius4-R2T2) runs on
this machine, kept for reference. It is not a benchmark suite. Do not test
performance aggressively: rerun these commands only when a change needs a
before-and-after comparison, such as a larger decoder cache.

Every number here was taken with other work running on the machine, so treat
it as a rough reference. Absolute times drift with the background load. Only
ratios between runs taken back to back, ideally interleaved, are reliable.

## Setup

- Mac17,7 (Apple M5 Max), macOS 27 build 26B123, measured 2026-11-12.
- All six models are fp16 and fully on the ANE: the encoder (8-second blocks),
  four decoder segments of 7 layers, and the head. The decoder takes chunks of
  T = 128 rows and has a 1024-row KV cache; the default is now 1152 (below).
- A heavy GPU job ran in the background throughout. ANE work does not compete
  for the GPU, but it shares memory bandwidth.
- `stream` forces Chinese and uses a 160 ms lookahead and 1 held-back token,
  with no context. tnt-asr holds back 2 tokens and adds a context.

## Inputs

- `zh.wav`: 19.5 s of synthesized Chinese with known text.
- `hank-55s.wav`: the first 55 s of a real English recording. 55 s is
  tnt-asr's take limit.

`say` and `afconvert` fail inside Claude Code's sandbox; run them outside it.

```sh
mkdir -p .tmp/bench && cd .tmp/bench
printf '%s' '今天下午三点，我们在会议室讨论了新产品的发布计划。市场部希望在十一月中旬上线，但是研发团队认为测试时间不够，至少还需要两周。最后大家同意先做一个小范围的内测，收集用户反馈以后再决定正式发布的日期。' > zh.txt
say -v 'Flo (Chinese (China mainland))' -f zh.txt -o zh.wav --data-format=LEI16@16000
afconvert -f WAVE -d LEI16@16000 -c 1 ../../../mlx-speech/outputs/source/raw/KOTH-Hank.m4a hank.wav
python3 -c "import wave; r = wave.open('hank.wav'); p = r.getparams(); w = wave.open('hank-55s.wav', 'wb'); w.setparams(p); w.writeframes(r.readframes(55 * p.framerate))"
```

## Commands

```sh
swift build -c release
pmset -g | grep powermode    # 0 Automatic, 1 Low Power, 2 High Power
B=.build/release/ane-speech A=artifacts/qwen3_asr
$B bench $A/decoder_segment0.aimodel
$B bench $A/decoder_head.aimodel
$B stream .tmp/bench/zh.wav --language Chinese --chunk-ms 640         # and 480
$B stream .tmp/bench/hank-55s.wav --language Chinese --chunk-ms 640   # and 480
```

## Results

One decoder pass, median of 20 runs:

| Pass | Low Power | Automatic | High Power |
| --- | --- | --- | --- |
| Segment (7 layers, T = 128) | 12.7 ms | 9.8 ms | 9.8 ms |
| Head | 5.7 ms | 4.3 ms | 4.3 ms |

Streaming, given as compute time over audio duration, then the median and
slowest window:

| Clip, window | Low Power | Automatic | High Power |
| --- | --- | --- | --- |
| zh 19.5 s, 640 ms | 0.42×, 261 / 418 ms | 0.31×, 197 / 314 ms | 0.32×, 198 / 316 ms |
| Hank 55 s, 640 ms | 0.42×, 257 / 524 ms | 0.32×, 195 / 404 ms | 0.32×, 196 / 402 ms |
| zh 19.5 s, 480 ms | 0.54×, 259 / 433 ms | 0.37×, 158 / 314 ms | 0.37×, 158 / 315 ms |
| Hank 55 s, 480 ms | 0.50×, 251 / 470 ms | 0.37×, 167 / 356 ms | 0.38×, 169 / 356 ms |

In every run, every window finished within its own length.

## Findings

- Low Power Mode slows ANE work by about 30%. High Power Mode gains nothing
  over Automatic.
- A take barely slows as it grows. The 55-second and 19.5-second clips run at
  the same ratio.
- Shorter windows cost more in total. A 480 ms window costs less than a 640 ms
  one, but not proportionally less, so 480 ms costs about 15% more compute
  overall.
- Accuracy on `zh.wav`, streamed through `serve` with tnt-asr's settings:
  480 ms and 640 ms windows give identical text. With 1 held-back token
  (`stream`), 480 ms mishears 但是 as 像是, while 640 ms gets it right; holding
  back tokens matters more than window length. Every streaming run mishears
  大家 as 再加. Offline transcription gets both words right.
- The 1024-row cache holds about 60 s of continuous speech: 34 template rows,
  13 audio rows per second, about 3 text rows per second, and up to 16 rows for
  generation. The cache size is a conversion choice, not a hardware limit
  (below).

## Decoder cache size

`--cache` on `reference/qwen3_asr_decoder_trace.py` sets the size; the runtime
reads it from `decoder.json`. To change it, evict the old decoder, re-trace
with the new size, and run `convert.qwen3_asr decoder`. tnt-asr ends a take
when the helper reports `full`, and otherwise at its own limit, which it sets
to 60 s.

Continuous speech fills about 16 rows a second, and about 50 rows are fixed, so
a cache holds roughly (rows − 50) / 16 seconds; pauses stretch that by about a
fifth. Sizes tried on 2026-11-12, all with the heavy background job still
running:

| Rows | Holds about | Cost per pass, interleaved |
| --- | --- | --- |
| 1024 | 60 s | — |
| 1152, the default | 69 s | 11–15% less than 1536, so a few percent over 1024 |
| 1536 | 90 s (a dense English take filled at 94.8 s) | +10–22% over 1024 |
| 2048 | 120 s | +42–47% over 1024 |

Every size tried compiles fully onto the ANE, and `stream-check` gives the
same transcripts at each.

### 2048 rows

- **It works.** The ANE compiler accepts 2048: all five decoder parts compile
  fully onto the ANE. Parity is unchanged from 1024. `stream-check` gives
  identical transcripts with tnt-asr's settings, and differs on the same single
  fp16 word with the default settings.
- **Every pass costs about 45% more**, whether or not the cache is full.
  Interleaved runs of one segment pass: 1024 rows took 16.9–17.6 ms, 2048 rows
  took 24.8–25.0 ms. Earlier, separate runs gave 12.7 ms against 18.1 ms.
- **Attention over the cache is the expensive part.** The extra cache adds only
  about 4% to the bytes a pass reads, yet about 45% of a 1024-row pass scales
  with cache length. A cheaper attention over the cache could speed up every
  pass, even at 1024 rows. This has not been investigated.
- **Streaming numbers were inconclusive.** Background load changed more than 2×
  during the session; the same 1024-row pass measured 12.7 ms and 27.6 ms. Under
  that load, 2048 ran at 0.74–1.27× real time on the 55 s clip and 1.53× on a
  115 s take.

## Qwen3-ASR, one take at a time

The official bf16 Qwen3-ASR 1.7B, converted the same way (`--checkpoint
qwen3_asr_1_7b`, cache 1152), transcribing a whole take once recording stops.
Measured 2026-11-12 in High Power Mode through tnt-asr's helper, with tnt-asr's
context:

| Take | Wait after stop, T = 128 only | With the one-row step |
| --- | --- | --- |
| zh 19.5 s | 2.9 s | 1.8 s |
| en 19.3 s | 3.3 s | 2.0 s |
| Hank 55 s | 7.9 s | 5.0 s |

Every generated token is one decoder pass, so the wait grows with the text,
not the audio: about 45 ms a token at T = 128 and 27 ms with the one-row step
(next section), plus about 0.3 s to encode and prefill a minute of audio. Low
Power Mode adds about 30%. Both clips came out
word for word right, including the two words R2T2's streaming mishears.

## Rows per decoder pass

A pass costs a fixed part (reading a segment's ~700 MB of weights and its KV
cache) plus a part that grows with its rows. Generating a token feeds one row,
but the T = 128 shape computes all 128. One-off measurement, 2026-11-12, High
Power Mode: decoder segment 0 of Qwen3-ASR, cache 1152, five shapes
interleaved for three rounds, each the median of 20 passes. All five compile
fully onto the ANE.

| Rows per pass | One segment | One token: four segments + head |
| --- | --- | --- |
| 128 (current) | 10.1 ms | 44.5 ms |
| 32 | 6.5 ms | 30.3 ms (−32%) |
| 16 | 6.1 ms | 28.8 ms (−35%) |
| 8 | 5.9 ms | 28.1 ms (−37%) |
| 1 | 5.7 ms | 27.2 ms (−39%) |

The floor is about 5.7 ms; below 32 rows each halving saves little. The head
stays at T = 128: it costs about 4.3 ms at any T.

Adopted: each decoder segment model now carries a second function, `step`, at
one row, and the runtime takes it whenever a pass feeds a single row. Both
functions share the weights on disk; Core AI compiles each separately, adding
about 2.7 GB of compiled cache per checkpoint. Parity is unchanged: every
check gives the same transcripts and the same per-step ids as before. R2T2
streaming at 480 ms, High Power Mode, median window: zh 158 → 127 ms, Hank
55 s 169 → 147 ms; it prefills most of each window at T = 128, so it gains
less than one-take Qwen3-ASR.
