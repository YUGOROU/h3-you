# Minimal Pi conversation / video director

This document describes the API. Real input, output, device, and Discord operations
remain on hold. Earlier finite-worker results in
E2E integration status are historical, not live-state evidence.

## Implemented

- `research/run_h3_director.py`: safe launcher, token injection, finite budget reservation.
- `research/director/agent.mjs`: existing Pi SDK (installed1.0.4, Node22+) with explicit resources/settings.
- `research/director/contract.mjs`: two custom tools and bounded proposal validation.
- `research/director/models.json`: HF Router, exact **Qwen/Qwen3.8-27B:cerebras**.

No new packages installed, no global Pi settings/extensions modified. SDK paths use
the current local Pi installation. No builtin read/bash/edit/write tools, MCP,
project/user instructions, discovered extensions/skills, compaction, automatic
retries, cache warming or persisted Pi sessions. This limits model capabilities;
it is not an OS sandbox for the trusted SDK process.

## Credentials and model routing

Launcher reads `~/.cache/huggingface/token` directly and injects `HF_TOKEN` only into
the child environment; never shell-expands it, places it in argv, prints it, or
writes it into models/auth/session files. Child stderr/provider payloads are not
forwarded; any exact token appearing in stdout is redacted. Debug/tracing and
NODE_OPTIONS are not inherited. Conversation results deliberately go to stdout;
do not redirect them to public logs.

Before paid execution, Router catalog must confirm the exact model + live
Cerebras route, tool support,65536 context and input/output pricing .99/1.49 per
million tokens. Request guard rejects model/provider fallback, output limits over
1024, or missing `reasoning_effort=none`.

Sources checked:
- https://huggingface.co/docs/inference-providers/providers/cerebras
- https://inference-docs.cerebras.ai/capabilities/reasoning

Cerebras Qwen3.8 defaults to high reasoning. First bounded test exhausted1024
output tokens with no proposal ($.01 bound). Using documented
`reasoning_effort=none`, the next test succeeded through Pi's native streaming and
tool handling:2 model calls,1689 input/89 output tokens, typed neutral/still/calm
reply. Another $.01 bound. No fallback or custom wire-protocol implementation.
These are smoke observations, not voice, lip-sync or conversational quality proof.

## Tools and state

- `h3_get_state`: fixed approved v12 geometry/reference and current turn/epoch.
- `h3_submit_turn`: stages one <=160-character spoken reply plus bounded
  expression (`neutral/gentle_smile/attentive`), motion (`still/subtle_nod`) and
  tone (`calm/friendly/thoughtful`). It does not apply them to any model or device.

Only successful turns enter an in-memory,8-turn dialogue history. Each turn uses
a fresh in-memory Pi session with that bounded history supplied as dialogue data.
One turn in flight; concurrent input is rejected, not silently queued. Cancellation
invalidates the active epoch, aborts Pi and rejects late drafts. Maximum3 model
calls/turn,1024 output tokens/call,60s turn deadline,1..8 turns/process and20min
process lifetime. Idle waiting makes no inference requests.

Legacy standalone v12 reference/identity/four forwards/geometry/visual prompt remain fixed.
Style values are **proposals only**. The tool process has no direct device/render
adapter; the existing unified conversation orchestrator owns downstream TTS/H3.
The explicit visual contract below is distinct from those legacy style hints.

## Explicit versioned visual intent route

`contract.mjs` now exports `VISUAL_SYSTEM`, `VISUAL_TURN_SCHEMA`,
`validateVisualIntent`, and explicit `validateDraft(value, {visualIntents:true})` /
`makeTools(Type, state, {visualIntents:true})`. The same two tools are retained;
no new agent, planner, action presets or privileges.

The visual draft is `{schema:'h3-director-visual-turn-v1', reply,
visual_intent:{schema:'h3-visual-intent-v1', instruction}}`. Instruction is natural
language (<=256 codepoints /1024 UTF-8 bytes), or null to **continue**, not reset
to a frontal pose. Unknown fields, completion claims and versions are rejected.
Tools still stage once, reject cancellation and report effects_executed=false.

The same existing entrypoint selects it only with explicit
`--headless-interrupt --visual-intents --vox-parent <reviewed-parent>`.
Without `--run`, no provider/inference/device operation occurs. With --run this
is a finite GPU operation requiring prior notification under existing authority;
real input/output holds remain. Default schema/SYSTEM/persona/model stay unchanged.
No separate visual runner exists.

The agent's private opt-in environment/READY, direct immutable spec/READY,
strict typed v2 payload, ordered epoch admission and native role/manifest binding
all match. Legacy consumers reject new proposals rather than silently stripping
intent. `resolve_visual_intent` runs only after current-turn admission. Speech
cancellation alone does not clear person/scene/action continuation. Intent text
is RAM/private runtime state; public structured receipts contain hashes only.

`visual_prompt` separates identity from frontal framing. The versioned native
role binding admits only its exact composed text; no arbitrary guard bypass.
Actual prompt/profile/cache hashes change, but the original normalized PCM/base
profile and ordered reference remain fixed. Both ranks verify actual prompt and
intent epoch before reporting native_conditioning_applied. visual_achievement
stays unverified, including after ACK.

New prompt keys are encoded with the SAME native conditioner, whole ordered
reference+text, never concatenated old/new embeddings. The explicit mode retains
that exact encoder at the cold encode-to-denoise release call site (sharded
encoders would otherwise be destroyed). It changes encoder lifetime/memory use,
not device/dtype/sharding, and does not override ordinary error/release cleanup.
Original DiT/decoder/weights persist. No reload/parking/quality/hardware fallback.
Added bank entries are capped at4; reference1024 and actual native token bounds
remain. Fresh encode timing is separate from cache reuse.

Dynamic visual mode explicitly uses the existing latest-tail policy, not periodic
first-tail anchors that could pull the camera back. Legacy headless mode retains
its existing every3 anchor selection. Both retain the original identity reference.

#### Stable selfie / no camera movement

The same attended entrypoint additionally accepts explicit `--stable-selfie`
with `--attended-visual --visual-intents` and the existing approved Comet/OBS/PCM
arguments. A device-free check is also available through the same CLI using
`--stable-selfie --visual-intents --vox-parent <reviewed-parent>` without attended,
interrupt or device flags; provide exactly two fresh typed turns. Its optional
`--review-av` uses a plain HeadlessAVSink archive, not interruption/paired cuts. Ordered references and the versioned visual transport are retained;
this is not a new runner. CLI selects the existing `conversation-tail-every3-v1`
anchor, while API callers must also set `conversation_anchor=True` explicitly.
The immutable session and READY bind `stable_selfie=true` plus that exact policy.
Missing, mismatched or interruption-mode selections are rejected before
execution; no silent fallback or change to the default dynamic mode occurs.

Pi uses a null-only visual instruction tool schema and a no-camera-movement
system policy. Non-null instructions are rejected again before publication,
worker TTS admission and native intent application. The backend starts with no
active movement and refuses inherited active movement. Null's legacy meaning is
unchanged. Ordinary generated listening and speaking motion remains; this does
not freeze or replay video. The first generated tail is reused every third global
window across both roles, with both-rank ancestry evidence verified by the
existing backend. Native quality, PCM, clocks and persistent decoder are unchanged.

This restores the **mechanism**, not demonstrated quality of the old single-image
99-second candidate under today's two-reference conditions. CPU coverage and
limitations: stable selfie implementation result.
Attended `--review-av` remains unsupported; device-free stable review is explicitly
separate and does not enable paired cuts. Stable visual events retain the actual
bounded ancestry selection digest; the collector requires a sequence3 anchor
application, not just a READY declaration.
Capture/output/cloud operations still require their existing concrete scope.
Generated-but-discarded native tail can differ from the last presented image;
internal continuity is not evidence of viewer continuity.

Native wiring result v2:
initial encoder-lifetime failure locally fixed; corrected finite two-turn run
applied both intents on both ranks and re-entered without H3 restart. Old stdout
EOF1.231ms, new decoded PCM19.849s. **This is not visual achievement or acceptable
conversation latency.** Content inspection still requires permitted artifacts;
do not retain private AV, extract frames, upload images or launch playback by
inference from this implementation. See a local status note for latest paused parent.

### Bounded review AV / paired suffix cut (Plan §0B)

On the SAME CLI, explicit `--review-av` additionally requires the visual/headless
route and binds `paired_cuts=true` in immutable spec and READY. Defaults still
clean temporary media. No new runner/player or automatic app launch.

The listening quarantine forward still runs. Its preceding verified native
commit/emit frontier, old source and next presentation generation are frozen
before the step. Full native packet hashes and full decoded frame count are
checked before RGB and S16LE are cropped at the same global boundary. The new
suffix uses `partially-presented-interrupted`, not full sink acceptance; FIFO
credit settles all verified bytes with an explicit omitted range. A later
interrupt cannot promote this suffix into the newer generation. New speech
head/all original PCM remain intact. This alone promises no4s latency removal.

A fresh campaign's `h3/receiver/review/` retains reaped headless encoder NUT spans
and a full native cut-reference MP4/S16LE with sidecars. The shared cap is2GiB,
original16-window bound is1654/24≈68.917s of unique media, headless≤600s under
the existing finite owner. Hardlinks avoid copying full spans before interrupt
control; copy fallback is explicit/timed. Temporary RGB/decode files still go.
Only after Job/Endpoint cleanup, bounded≤120s offline processing makes separate
MOV viewing renditions, checking frame counts and PCM. NUTs remain authoritative;
span MOV video is a labeled viewing transcode. No montage, time padding, upload,
external image inspection, browser, speaker or device operation. Sidecars retain
global origins/discontinuities and distinguish discarded prefix/reference from
submitted suffix. Buffered encoder bytes are not actual device playback;
decoded-read observation extents stay in result.json. Files remain for USER
review, not automatically deleted with the decoder temp directory.

CPU only:

```sh
node --test tests/test_h3_director_contract.mjs
python3 -m unittest discover -s tests -p test_h3_visual_intent_contract.py -v
~/.hf-cli/venv/bin/python -m unittest discover -s tests -p test_h3_visual_intent_wiring.py -v
~/.hf-cli/venv/bin/python -m unittest discover -s tests -p 'test_h3_review*.py' -v
```

These fixtures are neither real Pi inference nor visual success evidence.

## Historical standalone usage (not the unified visual route)

Offline SDK/config check (no credential read or network):

```bash
python3 -m research.run_h3_director
```

One explicit paid text turn, automatic environment-only credential injection:

```bash
python3 -m research.run_h3_director --run --text '短く挨拶してください'
```

Bounded JSONL background worker (stdio, not a network service):

```bash
python3 -m research.run_h3_director --run --serve --max-turns 8
```

Input examples:

```json
{"op":"turn","turn_id":"turn_1","text":"最近どう？"}
{"op":"cancel"}
```

Output types: `ready`, `turn` (proposal_only), `error`, `canceled`, `usage`, `budget`.
No daemon was left running after the original one-shot smoke test; a later
USER-authorized finite worker launch is described in the integration status. Starting the worker does not
start a microphone, camera, ASR, GPU job, player or Discord client.

## Historical standalone cost mechanism and tests

The following describes the standalone launcher's original accounting behavior,
not authority to query/change the USER-owned ledger or enforce a new budget loop.
The unified conversation's existing PiDirector does not perform those writes.

Shared locked HF ledger: reserve $.20 per allowed turn, based on worst-case
3*(65536*.99+1024*1.49)/1M=$.1992192. A clean exit reconciles full-price token
usage, conservatively rounded up to cents; unknown calls retain their per-call
bound, absent final usage retains the full reservation bound. Receipts contain
only model/budget/usage, no prompts or token. Bounds are not verified invoices.
No transfer from Modal; existing scoped HF cap remains190.35.

```bash
python3 -m unittest discover -s tests -p test_h3_director_runtime.py -v
node --test tests/test_h3_director_contract.mjs
```

Next: connect proposals to explicitly approved downstream adapters, beginning with
text-only turn scheduling. Preserve separate per-operation human approval for
sensors, audio routing, virtual camera and Discord sharing.
