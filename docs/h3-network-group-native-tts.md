# Native Vox Job → H3 Job via Network Groups

USER approved the hybrid architecture: Mac↔cloud WebRTC; native Vox L4 Job→H3 Job direct authenticated/encrypted TCP. Existing H3→Mac RTC output stays unchanged. The old all-hops-WebRTC requirement is superseded for this internal hop.

Status update: paired finite owner and ASR RTC integration are now implemented/CPU-tested; actual two-CPU-Job Group/TLS check passed. See hybrid guide and actual connectivity result. Native GPU acceptance is still pending.

## Implemented, not yet live-adopted

- `network_voxcpm.py`: native wire client over TLS1.3, exact immutable certificate DER SHA256 pin, owner key inside encrypted RPC. No HF bearer token is sent to the internal service. Certificate checked **before** text/key transmission.
- `network_voxcpm_service.py`: dedicated TLS service calling existing native Gate directly. Four bounded handlers; serial model owner, two generation requests, native hashes/sample counts/END and separate model-EOS evidence unchanged. No HTTP media handler, TURN, RTC fragmentation, resampling or codec conversion.
- `native_tts_backend.py`: explicit immutable `h3-native-voxcpm-network-v2` selector; original Endpoint v1 default remains untouched. Selects network client before any local-model preparation; key-only child secret for this transport; never falls back.
- `job_network.py`: verified REST `network={group,aliases}` fields, default-resource-group source identity checks. H3 launcher allows **only** the exact manifest-bound owned L4 source alongside H3 submission, rechecking before submit; all unrelated active Jobs still block rather than being canceled. Raw control-plane JSON is used because installed SDK1.7.2 drops network metadata.
- `hf_h3_voxcpm_network_worker.py`: dedicated native L4 worker, original eager runtime/model/reference pins and actual L4 check, independent finite expiry, source evidence. TLS material is Job-secret-only and ephemeral0600 files, never staged in source.
- `prepare_h3_network_tts_job.py`: offline source ZIP/payload builder; original image, L4×1, timeout1200s, attempts1, one per-run alias, no public/proxy media port. No SDK/root/model dependency upgrade.

**Job command must use the pinned image's own interpreter.** The L4 source runs on `pytorch/pytorch@sha256:417bd75d…`, which exposes `python` on PATH and has **no** `/opt/venv` (that path belongs to the custom H3 runtime image). The first real submission failed at container start with `exec: "/opt/venv/bin/python": stat /opt/venv/bin/python: no such file or directory`; the payload now uses `['python','-c',script]` and `tests/test_h3_network_voxcpm.py` pins that. Image, native pins, voice and watermark are unchanged.

Scheduling capacity is a separate failure mode: L4×1 was unavailable for `SCHEDULING` windows long enough to exhaust a 600s readiness budget, and a network-group-free `l4x1` canary behaved identically, so the group/TLS spec was not the cause. `run_h3_network_pair(..., scheduling_wait=...)` (CLI `--scheduling-wait`, ≤4h) waits inside that allowance while the source's own runtime lease stays clamped to 1200s from container start (`hf_h3_voxcpm_network_worker.lease`). Job `timeoutSeconds` is not consumed by queue time (an 180s-timeout canary survived >7min queued).

## Manifest and discovery

Network v2 carries only `schema,connection,session,boot,expires_at,pins,preroll`. Connection contains `group,alias,port,certificate_sha256,server_job_id`. Session/boot/expiry/native pins/preroll retain original semantics.

The H3 process resolves `${HF_NETWORK_GROUP_PREFIX}<alias>`; missing/invalid provider prefix fails closed. No arbitrary URL, public endpoint, DNS resolver substitution or HTTP fallback. Default resource group only in this implementation; adding an explicit non-default resource group requires binding it in both submissions and inspection rather than assuming compatibility.

Parent must stage a fresh group and TLS identity, verify exact source Job creation and placement, then build H3's immutable manifest using its actual source Job ID. Group membership alone is not GPU, model identity or encryption evidence. Source readiness occurs after model load; wait with finite readiness checks, never replay an uncertain stream request.

Placement for the TTS source is **not** country-gated: USER confirmed 2026-10-29 that native VoxCPM2 TTS has no region/territory restriction (license or preference), unlike H3. The approved gate therefore checks the non-geographic contract only (owned namespace, `l4x1`, pinned image, one fresh group/alias, `retry=0`, no public/media ports, exact owned Job identity) and records region as `unverified-region-unconstrained` instead of inventing evidence. H3 keeps its own unchanged territory/license gates.

TLS deliberately uses exact certificate pinning instead of public-CA/hostname validation, allowing fresh self-signed per-run identities. Native stream completion additionally requires TLS close_notify; abrupt TLS truncation is not accepted as trailing protocol EOF. Native protocol END still does not independently prove model natural EOS; source evidence must do that.

DNS/connect is not universally hard-cancellable. Server close cancels socket I/O and gate emission; it does not prove interruption of arbitrary running CUDA work. Independent Job timeout/cancel/terminal confirmation remains necessary.

## Placement compatibility and H3-first mode

Actual provider tests now reject both tested orders: L4-first Group cannot accept `h200x2`; original H200×2-first Group cannot accept `l4x1`. This is a provider placement constraint, not a reason to lengthen queue waits or weaken country/hardware gates. It does not prove universal incompatibility at every provider location. See H200-first result.

Explicit `run_h3_native_pair_headless --h3-first` now implements one actual H200 Job, authenticated single-use deferred capsule, real source-ID binding and adoption of that same Job (no second H3 allocation). Original image/pins/absolute expiry remain; L4 wait is clamped to H3 remaining time. Actual reservation/READY/rejection/cleanup succeeded, but capsule-to-model/AV remains unverified because L4 was rejected. Do not repeat without compatible placement evidence or an explicit architecture/hardware decision. Defaults remain unchanged, not an automatic fallback.

## Remaining before actual use

1. Use the implemented `run_h3_network_pair` / hybrid composition; the TTS-source placement gate is the non-geographic one above (no TTS region constraint), while H3 placement/notification rules are unchanged. Do not run the historical Endpoint owner as the hybrid owner.
2. Actual native L4/H200 stream acceptance has not happened yet: v2–v5 attempts ended in `SCHEDULING` timeouts or the `/opt/venv` container-start failure above (no GPU compute ran). Retry with a fresh private directory/group once L4 has capacity; do not replay an old artifact or group. See first GPU attempts.
3. Default-group connectivity itself is proven by separate CPU Jobs, not inferred from loopback.
4. ASR RTC single-replica public-sample WAN acceptance succeeded (20261030-v10); Endpoint paused. Do not repeat the successful gate. Native L4/H3 pair remains unaccepted: latest source6ae4002ec78863f9c62e2ad5 timed out readiness600s, CANCELED/cleanup confirmed, no H3 created. See latest result.
5. Separately approved attended fresh-speech/browser E2E. GPU ASR acceptance above does not authorize browser, capture or call operations.

## Sources

- https://huggingface.co/docs/hub/main/en/jobs-configuration#network-groups
- https://raw.githubusercontent.com/huggingface/huggingface_hub/main/src/huggingface_hub/_jobs_api.py (`_create_job_spec`, read-only reference, not runtime-downloaded code).

A read-only GET of our already canceled CPU Job confirmed raw inspect fields `retry=0`, `network=null`, `resourceGroupId=null`, `expose.ports`. It does not verify Network Group execution or placement.
