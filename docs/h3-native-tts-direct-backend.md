# Direct conversation: explicit native VoxCPM backend

## Status / safety boundary

CPU integration implemented. **USER explicitly lifted the actual H3 GPU hold on2026-10-29 for a short fresh two-turn Discord-free headless validation.** CPU success alone is not launch or device authority. Original local Irodori complete-WAV defaults and sentence opt-in remain unchanged. `research.run_h3_native_coupled` now owns a fresh finite Vox L4 Endpoint plus original H3 Job; existing Irodori Endpoint is not a Vox service.

Native mode borrows an independently authorized, authenticated, finite Vox service lease. Client close aborts local I/O; **it does not pause provider compute**. Resource deployment, coupled owner/expiry/pause confirmation and live WAN/H3 acceptance remain distinct evidence. The explicit coupled owner implements lifecycle; actual result/status is tracked locally, not implied by this API description. Never resume an old unknown-key lease or silently use local synthesis on failure.

## Immutable selection

Within `spec['live_direct_session']` only:

- `tts_mode: 'native_voxcpm'`
- `native_tts` exact fields:
  - `schema: 'h3-native-voxcpm-backend-v1'`
  - `url`: HTTPS `*.endpoints.huggingface.cloud`, no path/query/credentials/port. HF Jobs URLs and arbitrary relays are not accepted by this adapter.
  - `session`, `boot`:32lowercasehex service lease identities, intentionally independent of H3 conversation identity.
  - `expires_at`: actual service wall-clock expiry, no invented extension.
  - `pins`: exact `voxcpm_endpoint_protocol.PINS` (source/model/reference/rate/format/no-watermark).
  - `preroll`: exact fields `sample_rate`, `reserve_samples`, `max_reserve_samples`, `max_wait_seconds`. Initial candidate48000/48000/96000/2.0.

No secrets or extra fields are accepted. Native config is forbidden on legacy modes and at spec top level. Structural validation is clock-free; pre-submit rejects expired/insufficient(<120s) or >1200s leases, repeated immediately before actual submission. Runtime independently checks expiry and clamps direct-session admission/READY expiry to the TTS lease. This cannot guarantee enough time for an unmeasured H3 cold load; insufficient remaining runtime is refused.

Canonical SHA256 binds the complete validated config (including preroll) into direct READY. Parent verifies exact mode/digest and rejects a READY outliving the native lease. No ambient `H3_TTS_MODE` activation.

## Secrets / boot

Only child secret fields `H3_NATIVE_TTS_HF_TOKEN` and `H3_NATIVE_TTS_KEY` carry endpoint access. Parent validates them before runtime launch and injects them through the Job secrets map, never spec/argv/log files. Native factory consumes/removes both environment fields into the client. Operator must supply valid finite service metadata; no automatic token/key discovery or deployment.

`bootstrap_h3_fast_ref2va.prepare_session_tts` selects the native factory **before importing/scheduling local prepare_tts**. Native creates no TTS pool, installs/downloads/loads no Irodori model and starts no Irodori child. Direct native staging omits tts-manifest.json and local model/license/reference checks; H3's own assets/location/license/pins are unchanged. Legacy paths preserve their manifest and preparation checks.

Post-preparation bootstrap work is under an outer TTS owner, covering attention/import/download/hash failures. Late preparation completion is closed, executor shutdown always occurs, and normal run_session assumes ownership at handoff. Native runtime cleanup closes the remote client, not local close_tts_future/provider controls.

## Readiness / stream / conversion

Vox client has its own exact type-strict authenticated `/ready` validation, not inherited Irodori readiness. It requires pins/session/boot/expires_at/epoch0 before allowing `/stream`; complete-WAV synthesize is refused. Each turn reserves its epoch once. Stream validates READY→bounded native PCM→hash/count END plus clean HTTP body termination. Missing/malformed/trailing bytes, binding drift, replay or expiry fails closed with no fallback.

Connection object is registered before TCP/TLS connect; cancellation closes it, rejects a late connection before POST, and caps socket connect timeout at1s. **Python's synchronous DNS resolution is not interruptible or hard-bounded by that socket timeout**. No connector threads/retries were introduced. Before live acceptance, retain this cancellation limitation; do not claim a universal bounded DNS/worker-join guarantee.

Worker requests one full reply, streams into `relay_native_tts.produce` with the immutable preroll policy, and feeds existing growing-source admission. No external sentence splitting/completed-WAV slicing. Native48kfloat32→explicit48kS16 conversion has separate hashes/counts and saturation evidence; neither identity nor watermark is inferred. See [preroll contract](h3-native-tts-preroll.md).

Pending speech remains listening; native fixed-window readiness and preroll are max-like, not additive. Once speech starts, temporary missing PCM is cooperative wait, not silence/listening replacement. Protocol END seals source; it is **not independent native model EOS**. Preserve native decoder/common clocks/tail/credits, no rate change or extra output queue.

Private `native-tts-receipts.json` and post-join `h3_native_tts_receipts` stdout retain at most2turn receipts: epoch/source ID, pins, native and converted hashes/counts/formats, saturation, preroll decision/policy, protocol completion and false watermark/model-EOS labels. Receipt list is also returned by the worker. No reply text, credentials or new audio file is stored there.

## Explicit cancel v2 (worker/CLI CPU-wired; native verification pending)

`Gate(..., cancel_protocol=CANCEL_PROTOCOL)` and `RemoteVoxCPM(..., cancel_protocol=CANCEL_PROTOCOL)` negotiate `h3-voxcpm-cancel-v2`/stream content type v2. Default None preserves v1. The existing CLI selects it only with `--headless-interrupt --native-tts-cancel --vox-parent <reviewed-paused-parent>`. Its immutable native config is `h3-native-voxcpm-cancel-v4` plus `cancel_protocol`; direct session row/READY require `native_tts_cancel:true`. Lease stages a fresh immutable same-L4 runtime with the explicit HTTPS-only protocol env. v1 plus an extra field is rejected, not an activation shortcut. Default stays v1/drain; offline invocation starts no resources.

`request_cancel(epoch, deadline)` uses a separate bounded control connection with existing HF/app auth, session/boot/epoch binding and no media read lock. It dials only the active media TLS peer address with the original hostname/SNI/certificate validation (no DNS on control). No active peer means cancellation is unavailable; retain bounded drain/fail-closed policy. Requested/settling ACK is not a terminal. Repeated current-epoch requests are idempotent; old epochs cannot cancel a newer request.

Only the service's media owner closes the generator at a yield boundary, after preserving the whole yielded PCM prefix. v2 decoder validates cancelled terminal/count/hash/epoch/closed marker and EOF; client.last_completion appears only then. Close or transport failure is not successful cancel, and Python cleanup is not native cache-health proof. No next media request while terminal is unverified.

`relay_native_tts.produce(..., allow_interrupted=True)` additionally accepts the verified v2 interrupted terminal; default False rejects it. It retains the source as `PCMIngress` state `interrupted`, not sealed/full utterance. `finish_interrupted` binds exact prefix count/hash, refuses append/complete_source, retains existing tail reads and original finite deadline, and treats missing suffix as wait (not EOS/padding). Source owner must retire at the H3 safe boundary and cancel only after tail use. Zero-prefix interruption remains unpublished. The opt-in worker now selects this bridge and owns even unpublished/zero prefixes. Separate max2 control queue/thread avoids blocking media/command ingestion. Control≤5s, verified terminal/fallback≤15s within worker deadline; missing terminal fails closed, never overlaps the next request. Source retention ends only through native safe retirement/tail cleanup. All worker threads are joined. Legacy default remains drain.

Implementation status/tests and actual-model proof requirements: worker wiring CPU report; prior foundation report. Actual native attempts both reached old naturalEOS before the Mac-voice trigger, so no actual cancel was sent. Normal v2 streaming/two naturalEOS requests and full new sink PCM were verified; cancellation/WAN control/cancelled-cache reuse and causal latency improvement remain unverified. Fix deterministic producer-state trigger before further native tests. Visual-return content remains separately unreviewed.

## Offline usage / verification

```
python3 -m research.launch_h3_direct_conversation \
  --directory FRESH_PRIVATE_SESSION \
  --native-tts-config PRIVATE_NATIVE_LEASE_CONFIG.json
```

Without `--run`, this only validates metadata (and an existing staged deployment if present), requires no secrets, makes no GPU/cloud/device request and does not create a new session directory. `--native-tts-config` is mutually exclusive with sentence mode. Resource/call/device permissions remain operation-specific.

## Coupled finite owner (explicit run only)

`python -m research.run_h3_native_coupled` is offline-only by default. `--run --directory FRESH_PRIVATE_ROOT` is the explicit user-approved headless GPU operation, not a device/call route. It serializes three TTS owner locks, checks fresh Job/Endpoint inventory, and refuses unresolved native/Irodori/Vox fences. Original approved isolated lock hashes/model inventory/private reference and fresh AWSus-east-1 L4 catalog/quota are checked before creation. No accounting queries/ledger writes.

Exactly one authenticated L4 replica, immutable source/model/ref/eager runtime, fresh service boot/session/key,1200s absolute expiry shared across uv re-exec; one create POST with durable uncertain-create fence. Entry validates actual1L4/Torch2.8 and logs bounded runtime/completion metadata, never texts/media. H3 remains original pinned H200×2 Job with its existing placement/license gates and attempts1/finite timeout.

The frozen native config requires Endpoint identity before H3 submission; deferred L4 activation after H3 READY is **not implemented**. Owner requires600s lease remaining before H3 staging and180s at receiver sink admission. These checks do not predict/guarantee cold-load duration and never extend expiry.

Authenticated `/evidence` returns at most2native completions with exact native-float hashes/counts and actual runtime facts. Protocol END still labels model EOS false; owner separately requires hash-bound observed natural stop+iterator exhaustion. Worker receipt stdout is collected before Job cancellation (up to15s polling, SDK network calls have their own timeouts; not a universal DNS/read hard bound). Exact speech regions in locally captured paired PCM are compared against converted hashes/counts, with final sink/tail required. Neither hashes nor EOS prove identity, complete spoken words or audible playback.

Owner always attempts provider pause even after an uncertain create. Diagnostics cannot skip independent H3 cancellation/terminal confirmation. Endpoint fence clears only on confirmed pause or explicit create rejection. Local saved AV is bounded32MiB; no generated media upload. Finite owner alarm1500s triggers cleanup but Python DNS remains the stated interruption limitation. No implicit retry, model/precision/default/rate change, OBS/BlackHole or Discord.

CPU coverage/result: connection report. CPU driver tests use fake native frames/pump; they are not CUDA quality, synchronized real video, sink ACK or audible playback acceptance.
