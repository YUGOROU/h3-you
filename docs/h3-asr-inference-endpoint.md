# Finite T4 ASR Inference Endpoint

## USER acceptance (latest)

**ASR marked SOLVED / accepted for this application at USER request.** Remaining
word errors are handled by Pi contextual correction, normally without asking for
repetition. This supersedes earlier wordaccuracy/semanticcorrection blockers
below, but not genuine-final, full-input coverage, cleanup or budget safeguards.
Pi SYSTEM and live turn prompt updated; historical measurements/errors remain
honestly recorded, not claims of perfect ASR. Next acceptance target is H3-inclusive
video-call E2E (`docs/h3-director-e2e-status.md`).

## Latest H3-free retest: whole question, recovered float32 boundary

USER requested retest excluding H3; scoped explicitly to Discord→ASR→Pi text,
no Irodori/voice return. Scheduled10:56:02JST capture started10:56:02.001107.
Resumed still-valid originalcampaign/keyrotation/deadline, no renewal or extra
.75 reservation. That deadline is now expired; do not resume it again.

Native final contained whole question「こんにちは日本で最も多くの授業員を抱える企業は何ですか」
@16.243s. Empty final flush@47.824s reported44.98099899291992 vs sent44.981s.
Both correspond to719696 PCM samples. Exact floating-point comparison wrongly
prevented completion and timed out. Adapter and delivery now compare rounded16k
sample counts (half-sample representation tolerance, not missing-chunk tolerance).
40 unique offline tests pass, including socketpair float32 terminal completion.

Original failed receipt preserved at P/hf-asr-attended-20261020-whole-retest1/result.json.
Recovered genuine finals using recorded commit+terminal sample coverage, SHA-bound
in verified-coverage-pi-recovery.json; no partial promotion/re-recording. Exact Pi
succeeded, but asked if「授業員」means正社員: contextual correction did NOT pass;
従業員 is not limited to正社員. No web-verified company ranking, no H3/TTS output,
no claim of clean single-owner live execution after last float fix. Full-question
transport demonstrated with diagnosed recovery, recognition/semantics remain work.

After recovery APIverified EndpointPAUSED/noactiveHFJobs. Retest additional bound
.01Pi only; shared originalASR.75. Current total216.505600/cap216.763374/remain.257774.

## Whole-input fix after USER transcript confirmation

USER's intended utterance was「こんにちは! 日本で最も多くの従業員を抱える企業は何ですか」.
Only initial segment reached Pi: old first-final policy ended capture prematurely.
Thus earlier transport proof below is **partial-utterance only**, not successful
whole-question recognition.

Owner now explicitly selects `completion_policy='input-finalized'`: keep sending
until actual input EOF/45s bound, commit, collect native completed segments in
order, require terminal native finalized coverage >= ALL sent PCM before setting
turn final. Empty terminal completion contributes coverage, not text. Partials,
EOF, deadline, or early final alone never complete a whole turn. Repeated spoken
segments are preserved. Legacy adapter default first-final is unchanged for
compatibility, but must not be used by attended owner. Commit send/state and
receiver checks share a lock to avoid acknowledging an unsent commit.

Delivery guard verifies aggregate equals all genuine segments, plus commit and
coverage, rather than comparing only last segment. Owner timeout60s includes45s
input and bounded flush. This is **bounded-window completion**, not a natural
end-of-utterance detector or realtime acceptance.38 unique offline tests passed,
including local socketpair multi-segment/finalized-coverage regressions. No GPU,
new capture, paid inference, or live whole-question acceptance for this fix.

## Latest attended result: 2026-10-20 (before whole-input fix)

**Live Comet/Discord → native Endpoint initial segment → exact Pi passed. Whole question was truncated. No H3 AV or
remote output acceptance.** Scheduled capture start10:31:05JST, actual start
10:31:05.001783; USER asked to speak10:31:10–10:31:40. Endpoint ready before time.
Native nonempty final after9.6s sent, errors[], transport/CoreAudio cleanup true,
capture stopped before Pi. Comet tap still includes all tabs, not isolated Discord.

Owner initially failed Pi reservation: required.20, not assumed.01. After own
terminated H3 bound reconciliation, same fresh final was passed by stdin to exact
Qwen/Qwen3.8-27B:cerebras, 2calls, successful clarification proposal only. Original
owner failure receipt preserved; separate `P/hf-asr-attended-20261020-v1/pi-recovery.json`.
Do not claim the owner ran end-to-end unchanged or reuse this text as future
post-READY input. H3 was cancelled during initialization because our monitor
incorrectly counted scheduling against its READY wait; not a model crash.

New owner options: `--capture-not-before` future UNIX time <=600s; late readiness
>3s aborts capture, finite wait and UTC receipts. `--parent-campaign` validates
paused prior deployment/current private branch, then creates fresh deadline/key
and .75 reservation (not an expired lease reuse). Source upload may retain the
same immutable commit when reviewed source bytes are unchanged.32 unique CPU
adapter/owner/nativepolicy/Endpoint tests passed; scheduled path live-tested.

H3CANCELED/ASRPAUSED/no other active owned HF Jobs API verified after recovery.
Budget/latest blockers: `docs/h3-compute-budget.md`, a local status note. Earlier results
below describe previous sessions and do not override this result.

## Verified result

USER approved implementation/validation after proposing dedicated HF Inference
Endpoints only during demos: T4x1 per replica, min1/max3. This overrides the prior
no-GPU-ASR preference **for this finite ASR validation/demo scope only**. Not an
H3/TTS provider/quality change or open-ended GPU lease.

Created **one Endpoint** `YUGOROU/h3-asr-t4-7c926aae18`, not three Endpoints.
AWSus-east-1/nvidia-t4x1, min1/max3, $0.50/hr per active replica. Now **PAUSED**,
API-confirmed after both English and Japanese sessions and again at handoff.
CPU Space remains paused. No personal capture/upload, local model inference,
H3GPU job or automatic background watcher.

True input streaming confirmed: authenticated WebSocket handshake, realtime-paced
160ms PCM16 chunks, partials during input, native utterance finals. CUDA readiness
verified (`device:cuda`); no CPU fallback. HF control-plane requests T4; no separate
runtime GPUname/SM telemetry was collected. Anonymous `/ready` denied.

| Public input | First nonempty partial | Last final | Progress lag at end |
|---|---:|---:|---:|
| JFK11s English |1.362s|11.484s|0.394s|
| JFK repeated3x33s |1.371s|33.483s|0.324s|
| FLEURS11.1s Japanese |2.982s|10.499s|0.293s|
| Same Japanese repeated3x33.3s |2.939s|32.542s|0.311s|

Times start after WebSocket connection. Input includes1s appended silence. Japanese
WAV itself has trailing silence: lastfinal before WAVend is **not negative EOS
latency**. Actual end-of-speech latency was not isolated. A wall/audio progress
ratio near1 in a realtime-paced test is expected, **not unpaced kernel RTF or
GPU utilization**. No growing multi-second CPU-Basic-style backlog in these short
sessions. Not minute-scale sustained/concurrent/load-scale acceptance.

Japanese reference:
> 多くの場合、海外のギャップイヤーコースに入学することで、帰国後に実際に大学に進学しやすくなります。

First run output included `横の場合` / `ギャットイヤー` / `企画後` instead of
`多くの場合` / `ギャップイヤー` / `帰国後`. NFKC/whitespace/punctuation-normalized
CER: **10.87%** for one sample, **7.25%** across its three repetitions. This is
one sentence/repeated audio, not a corpus-wide Japanese accuracy estimate. First
partial~3s and recognition mistakes still need optimization/acceptance; do NOT
call this complete realtime Discord readiness. No Comet→Pi→AV integration yet.

## Attended Comet integration (executed; no live ASR text)

Latest attended attempt: HFH3Job`6ad62f7f0a35baab53a05fc5` reachedREADY with205s
remaining; THEN Endpointstartup~60s and45sCometPCM capture completed cleanly.
719696frames/44.981s,115071nonzerosamples/peak32678/RMS2188.59; all44.981s sent/
committed. Onlyempty nativecompleted event at49.249s; no nonempty nativefinal,
so **NO Pi, NO AV generated/delivered**. H3CANCELEDandASRPAUSEDverified; noother
ownedHFJobsactive. CurrentHFbound214.163374/remaining2.449963, fullH3.1.833337+
ASR.75retained, noPi charge. NoDiscordE2Eacceptance.

ControlledPUBLICFLEURSJapanese directinput into SAMEadapter/deployedrevision
subsequentlyreturnedgenuinefinal/errors[]/cleanuptrue, pausedagain, withinSAME
originalcampaign.75reservation/deadline. ASRmodel/transportworks; liveinput
content/timing/Discord-onlysourceunproven. WeannouncedspeechBEFORE60sASRcoldstart,
so utterancemayhavefallenoutsideactual45swindow. DoNOTassertthatastheprovensole
cause; nonzero signalcouldincludesoundeffects/otherComettabs. Fixready/start-time
coordinationbeforeanotherfullGPUattempt, andfirstverifyreceivedspeech. Public
samplewasnotusedasreplacementPi/H3liveinput. Localadapteremptyflushedcompleted
nowreportsno_native_finalratherthanwaitingtimeout; CPUtested/notlive-retested.

Receipts: P/live-voice-session-20261019-hf-endpoint1/{submission,ready-observed,
terminal}.json and P/hf-asr-attended-20261019-v1/{result,public-diagnostic}.json,
capture-metadata.jsonl (energy/counts only), no raw personalaudiofile.

### Implementation and preflight context (prior to attempt)

Latest USER permits Discord connection/GPU and explicitly selects **HF for
H3+Irodori too**, superseding old Modal-only instructions. The previous request
for Modaladditional2.60 is withdrawn. H3HF660s/H200x2 reserves1.833337; freshASR.75
andPi.01 fit currentHFremaining5.033300 without increased cap. FreshHFsession
`P/live-voice-session-20261019-hf-endpoint1/deployment` prepared but NOT submitted,
uploaded or reserved; no newGPU running. Nativeendpointpolicy and
`research/deliver_h3_endpoint_turn.py` implement truthful cloudASR labeling and
one freshpostREADY→HFinputseq1→newhash-boundAV→existingOBSUDP. Helper cancels only
currentownedHFJob in finally; does not claim remote reception/realtime.28unique
CPUtestsPASS acrossadapter/owner/nativepolicy/Endpoint suites.

Echo prevention approved: `research/mute_h3_builtin_output.swift --mute` confirmed
built-inspeakers already muted before/after and both globaldefaultsBH2chunchanged.
Comet has2Discordtabs but AppleEventsJavaScriptexecutiondisabled; do NOT enable
thatsecuritysetting silently. AXtree exposed browserchrome only, no Discordroute
controls; no settings/join performed. USER asked to select DiscordinputBH2ch,
outputmutedMacBookAirspeakers and join target call. Wait confirmation before paid
lease; actualDiscordoutput/callstate unverified, not proof from savedsettings. New
`research/run_h3_endpoint_voice.py` defaults to an offline plan; explicit
`--run --capture-comet` stages NEW immutable reviewed source/deadline/ownerkey and
$0.75 reservation on the SAME paused T4 Endpoint. Original model/archive pins
remain unchanged. Never replays expired public-validation artifacts. NativeWS
adapter `research/h3_personal/endpoint_asr.py` starts caller-provided capture only
AFTER authenticated transport/session setup; bounded first nonempty native final
ends one utterance. No speculative Pi calls. Input pacing does not catch up in
bursts; commit follows last frame duration. Rawaudio remains pipes only.

`research/stream_h3_comet_pcm.swift`: noSpeech/mic/default routing; private
application-process tap, bounded32-bufferqueue, conversion16kmonoPCM16LE,
nonblockingstdout, diagnostics-onlystderr, max45s and signal cleanup. Includes
ALL Comet tabs and matched subprocesses, NOT Discord-isolated. Stop otherComet
audio before any attended upload. Caller confirms nonzero captured/transmitted
samples and cleanup; stops capture BEFORE Pi/any future outboundAV. Pi gets text
via stdin, not argv, exactQwen/two-tools/no-fallback. NormalSIGTERM maps to owner
cleanup/pause; SIGKILL/hostloss still require manual allocation shutdown check.

UnsignedSwift6strictconcurrency/warnings-as-errors compile PASS; enumeration-only
--check found3Cometprocesses. Eightadapter+fourowner+eightEndpointCPUtestsPASS.
No new livecapture/ASR/Pi/GPU test, Discordjoin or device-route change executed.
Owner only covers Comet→ASR→Pi; H3/OBS integration remains separate, not claimed.

Historical Modal detour (NOT a current blocker; USER corrected to HF above):
Modal ledger has2.12790875931802708447200 remaining; smallest jp/256GiB/
600s allocation reserves4.72 (19.799136/hr + finite startup/shutdown/buildmargin),
so needs **additional2.60** dollar authority. USER GPUok is not an explicit cap
increase. No HF→Modal fund transfer. Even a minimum allocation is not a guarantee
that cold-start plus full live test fits; require readiness/window gate.

Readonly inventory confirms BlackHole2ch is still BOTH Mac defaults; no second
virtualbus available. If Discordalso uses BH for received output and sent input,
it returns incoming audio directly to caller (even with Comet-only ASR tap).
Before live connection isolate received Discordoutput, e.g. app-only output to
explicitly muted built-in speakers, leaving globalBHdefaults unchanged; obtain
per-operation approval (now obtained) and verify no physical sound. Do not silently install a
new virtualdriver/change defaults. Discordchannel/join state not verified:
AppleScript metadata query timed out; no UI capture or securitysetting workaround.

## Implementation

- `research/run_h3_asr_endpoint.py`: offline by default; `--run` explicitly stages
  private runtime repo, verifies namespace quota/rate, reserves existing budget,
  creates one Endpoint, performs public English11s/33s sessions and pauses.
- `research/validate_h3_asr_endpoint_japanese.py`: offline by default; explicit
  `--run` can resume SAME paused revision only within ORIGINAL deadline/reservation,
  rotate owner stream key, test exact public Japanese file and pause. No new
  deadline, reservation, model, precision, provider or endpoint.
- `research/endpoints/nemotron_t4/{entrypoint,gate}.py`: custom container runtime,
  native websocket relay/no bodies logged or saved. ExplicitCUDA, batchingoff,
  statearena1, nativeHTTPworkers2, noUI/capture/TTS. Runtime max1200s plus shared
  absolute deadline; foreground control-plane caller MUST pause allocation.
- `research/probe_h3_asr_cpu_space.py`: shared bounded RFC6455 client, owner-header
  allowlist, native incremental-input protocol and language prompt. Completion
  waits for full audio_processed coverage, not merely first post-commit segment.
- Eight new Endpoint CPUtests plus seven CPU Space/protocol tests PASS. These are
  security/accounting/protocol contracts, not substitutes for GPU observations.

Local post-test hardening: gate permits one stream per replica with a semaphore;
client explicitly reports timeout/incomplete processed OR finalized coverage/
surviving-reader errors. Cleanup now attempts pause BEFORE any lookup and retries
transient failures, always persisting an unconfirmed terminal receipt on failure;
resume verifies provider/access/hardware/runtime/deadline/ledger reservation and
rejects unexpected replica schedules. These latest changes are CPUtested but **not redeployed/retested on GPU**. Existing
immutable deployed revision remains original successful version. Recorded raw
GPU events independently prove complete audio_processed coverage for all4sessions.

### Pins / artifact packaging

No Docker daemon/build/install needed. Endpoint builds its artifact image from a
pinned public Python base plus private HF model-repo snapshot, entrypoint override
`python /repository/entrypoint.py`. Source and binaries are uploaded privately;
no HF token, runtime key or public audio is baked into the repo/image.

- Python3.12-slim-bookworm digest:
  `34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258`.
- Private repo `YUGOROU/h3-asr-t4-runtime-7c926aae18`, commit
  `d1c8ce9b28361a515df91c047dbd3554d2baaff1`.
- Official NeMo-Speech.cppv0.2.0LinuxCUDA archive105010050bytes, SHA256
  `aef573fcc6c2c759401d2ed03e5ca9e401ed23b0e38cb852d917080c1e2d4515`.
  Official build includes75-real(T4) as well as other architectures. Bundled
  libraries/notices kept intact. Artifact downloaded and independently hashed.
- Original NVIDIA Nemotron3.5ASR Q8_0 GGUF741548352bytes, revision
  `1c8deaecc64b91f034d73e08dd8b64625eb3395d`, SHA256
  `a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae`.
  Same ASR-only experimental precision as CPU Basic; not nativeNeMo/BF16 parity
  or a precision change to H3/nativeTTS. ModelOpenMDW1.1/runtimeApache2+notices.
- Public Japanese `google/fleurs`, revision
  `70bb2e84b976b7e960aa89f1c648e09c59f894dd`, ja_jp/validation/row0/id1519,
  `10009621509886822059.wav`, CC-BY4.0 confirmed from pinned dataset metadata.
  Source16kmonoPCM32/177600frames; SHA256
  `9627b8f5d46806e1a433b443d692fdfb24f26a35c1a47eba26291475fc643d7b`.
  FFmpeg converts public file to PCM16 in memory, no input devices. Datasetviewer
  test split returned scanlimit500; validation split worked. Not model failure.

Private receipts/audio stay outside project at
`$H3_PRIVATE_ROOT/hf-asr-endpoint-20261019-v1/`:
`preflight.json`, `intent.json`, `submission.json`, `result.json`,
`japanese-public-receipt.json`, `japanese-result.json`, `terminal-confirmation.json`.
Intent excludes secret; runtime owner key only in memory and encrypted Endpoint
secrets, rotated for second session. No broad HF credential forwarded into native
ASR process request logs. HF outer authenticated gateway + extra owner-key header
protect the stream. `/ready` relies on HF auth. `private` Endpoint means PrivateLink,
not private Space semantics; this uses authenticated HTTPS/WSS instead.

## Budget, shutdown, and next demo

Namespace-scoped provider API `/v2/provider/YUGOROU` confirmed AWS T4 quota30/used0
before launch ($0.50/hr); public `/v2/provider` misleadingly reports zero. Never
use public inventory quota as an account permission verdict.

Reserved **$0.75** = max3replicas×$0.50/hr×30min, including initialization/idle/
English and Japanese resumes/shutdown allowance. The immutable original absolute
`deadline` in `submission.json` remains authority; it was not extended.
The complete reservation remains conservatively counted, not an invoice or active
lease: HFbound **211.580037 / cap216.613337 / remaining5.033300**. No reset/addition
of reported$15.45 balance. General HF billing docs make Endpoints compute-credit
eligible; exact displayed balance bucket/debit not independently reconciled.

An internal server timeout is NOT allocation termination: pause verified by API
is mandatory. Foreground orchestration pause-in-finally cannot protect against
complete host/network failure; if interrupted inspect/pause the owned Endpoint
immediately. No automatic restart/resume, watcher, perpetual min1 or scale-to-zero
substitution. Future demo needs explicit finite scope, fresh deadline/snapshot/
owner key, reservation and readiness checks; do not blindly resume expired
artifacts/replay historical submitters. Max3 scales concurrency, not single-stream
latency; separate connections may reach different caches. No transparent ASR-state
migration/reconnect/scale-down safety is implemented.

References:
- https://huggingface.co/docs/inference-endpoints/pricing
- https://huggingface.co/docs/inference-endpoints/guides/custom_container
- https://huggingface.co/docs/inference-endpoints/security
- https://huggingface.co/docs/hub/billing
- https://huggingface.co/blog/s2s_endpoint
