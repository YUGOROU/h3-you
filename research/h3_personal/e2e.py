"""Private sequential H3 demo (not true causal streaming).

Run: python3 -m research.h3_personal.e2e --help
Requires ffmpeg/ffprobe; local TTS: pip install voxcpm soundfile (Python 3.10-3.12).
Use an already-running H3 Ref2VA service; this script launches no GPU jobs.
Example:
  python3 -m research.h3_personal.e2e --llm-model MODEL \
    --reference-image portrait.png --reference-audio voice.wav --output private-demo

Contract checked against vllm-project/vllm-omni source (research pin:
47776fd350f55f21e016d58167129476497cb3e5):
recipes/MiniMaxAI/MiniMax-H3.md; entrypoints/openai/api_server.py,
video/generation/helpers.py, protocol/videos.py; diffusion/models/minimax_h3/
pipeline_minimax_h3.py. Sync uses form fields, typed image_url/audio_url data
URIs, raw video/mp4 plus X-Inference-Time-S, not JSON completion URLs.
VoxCPM API: OpenBMB/VoxCPM README (reference_wav_path, tts_model.sample_rate).
Personal LoRA compatibility is NOT established; no adapter is requested.
Server-side preloaded/fused adapters cannot be verified by this client.
"""
from __future__ import annotations

import argparse
import base64
import functools
import ipaddress
import json
import math
import os
from pathlib import Path
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, HTTPRedirectHandler, ProxyHandler, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Redirect refused: personal data must not change endpoints")


def check_endpoint(url: str, allow_remote: bool = False) -> None:
    p = urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname or p.username or p.password or p.fragment:
        raise ValueError("Expected HTTP(S) endpoint without credentials/fragment")
    # Literal loopback only: no DNS rebinding through arbitrary names.
    try:
        local = ipaddress.ip_address(p.hostname).is_loopback
    except ValueError:
        local = p.hostname.lower() == "localhost"
    if not local and not allow_remote:
        raise ValueError("Remote personal-data endpoint requires --allow-remote-personal-data")


def request(url, body, content_type, *, key="", timeout=14400, allow_remote=False):
    check_endpoint(url, allow_remote)
    headers = {"Content-Type": content_type}
    if key:
        headers["Authorization"] = "Bearer " + key
    # Ignore environment proxies; never follow a redirect with personal assets.
    opener = build_opener(ProxyHandler({}), NoRedirect())
    with opener.open(Request(url, data=body, headers=headers), timeout=timeout) as r:
        return r.read(), dict(r.headers.items())


def data_uri(path: Path, mime: str, max_bytes: int) -> str:
    data = path.read_bytes()
    if not data or len(data) > max_bytes:
        raise ValueError(f"Invalid reference size: {path}")
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def run_command(args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def probe(path):
    return json.loads(run_command(["ffprobe", "-v", "error", "-show_streams",
                                   "-show_format", "-of", "json", str(path)]))


def stream_duration(stream):
    value = float(stream["duration"])
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Invalid media duration")
    return value


def duration_for_tts(seconds):
    if not math.isfinite(seconds) or seconds <= 0 or seconds > 15:
        raise ValueError("TTS must be positive and <=15s; refusing truncation. Ask agent for shorter speech.")
    return min(15.0, max(5.0, math.ceil((seconds + 0.25) * 24) / 24))


class LocalVoxCPM2:
    def __init__(self, model_path, reference):
        from voxcpm import VoxCPM
        import soundfile
        self.sf = soundfile
        self.model = VoxCPM.from_pretrained(model_path, load_denoiser=False)
        self.reference = str(reference)

    def synthesize(self, text, output):
        wav = self.model.generate(text=text, reference_wav_path=self.reference,
                                  cfg_value=2.0, inference_timesteps=10)
        self.sf.write(str(output), wav, self.model.tts_model.sample_rate, subtype="PCM_16")


def agent_utterance(args, history, turn):
    messages = [{"role": "system", "content":
                 "You are an autonomous conversational agent in an AI avatar demo. "
                 "Choose your own topic and what to say next. Return only one short spoken "
                 "utterance, about 5-10 seconds, never more than 12 seconds. No stage directions. "
                 "Do not pretend to be the real person. " + args.agent_instruction}]
    messages += history
    messages.append({"role": "user", "content": f"Choose and speak turn {turn + 1} of 3; build on prior turns."})
    payload = {"model": args.llm_model, "messages": messages, "max_tokens": 160,
               "temperature": 0.8, "stream": False}
    raw, _ = request(args.llm_url, json.dumps(payload).encode(), "application/json",
                     key=os.environ.get("H3_LLM_API_KEY", ""), timeout=args.timeout,
                     allow_remote=args.allow_remote_personal_data)
    result = json.loads(raw)["choices"][0]
    text = result["message"]["content"]
    if result.get("finish_reason") != "stop" or not isinstance(text, str) or not text.strip():
        raise ValueError("LLM did not return a complete spoken utterance")
    text = text.strip()
    history.extend([messages[-1], {"role": "assistant", "content": text}])
    return text


def generate_h3(args, image_uri, audio_path, duration, text, output, turn):
    fields = {"model": args.h3_model, "prompt":
              "The person in the reference image speaks naturally, lip-syncing precisely to "
              "the complete driving audio, then rests silently. AI-generated avatar. Speech: " + text,
              "width": str(getattr(args, "width", 768)),
              "height": str(getattr(args, "height", 1344)), "fps": "24",
              "num_inference_steps": "50", "flow_shift": "12", "seed": str(3101 + turn),
              "image_reference": json.dumps({"image_url": image_uri}),
              "audio_reference": json.dumps({"audio_url": data_uri(audio_path, "audio/wav", 15 * 1024**2)}),
              "extra_params": json.dumps({"task": "ref2va", "duration": duration,
                                           "audio_mode": "lock_source", "audio_flow_shift": 3.0})}
    start = time.monotonic()
    raw, headers = request(args.h3_url, urlencode(fields).encode(),
                           "application/x-www-form-urlencoded",
                           key=os.environ.get("H3_VIDEO_API_KEY", ""), timeout=args.timeout,
                           allow_remote=args.allow_remote_personal_data)
    headers = {k.lower(): v for k, v in headers.items()}
    if headers.get("content-type", "").split(";")[0].strip() != "video/mp4" or not raw:
        raise ValueError("H3 sync must return raw video/mp4, not JSON/base64/job URL")
    output.write_bytes(raw)
    return {"client_seconds": time.monotonic() - start,
            "server_seconds": headers.get("x-inference-time-s"),
            "request_id": headers.get("x-request-id")}


def validate_av(path, target, tolerance=0.15):
    info = probe(path)
    durations = []
    for kind in ("video", "audio"):
        streams = [s for s in info["streams"] if s["codec_type"] == kind]
        if len(streams) != 1:
            raise ValueError(f"Expected one {kind} stream in {path}")
        durations.append(stream_duration(streams[0]))
        if abs(float(streams[0].get("start_time", 0))) > tolerance:
            raise ValueError("Nonzero A/V start offset")
    if any(abs(d - target) > tolerance for d in durations) or abs(durations[0] - durations[1]) > tolerance:
        raise ValueError("A/V duration mismatch")
    run_command(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"])
    return durations


def mux_original(video, original, output, target, spoken_seconds):
    if not 0 < spoken_seconds <= target <= 15 or target < 5:
        raise ValueError("Mux duration would truncate TTS or violate 5-15s limits")
    streams = [s for s in probe(video)["streams"] if s["codec_type"] == "video"]
    if len(streams) != 1:
        raise ValueError("H3 returned no unique video stream")
    actual = stream_duration(streams[0])
    if actual < spoken_seconds or abs(actual - target) > 0.25:
        raise ValueError("H3 video cannot cover complete TTS or requested duration")
    # H3 frame quantization may differ slightly. Only tail silence/frozen idle
    # may be trimmed/padded; speech is never shortened, and H3 audio is discarded.
    run_command(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-i", str(original),
                 "-map", "0:v:0", "-map", "1:a:0", "-vf",
                 "setpts=PTS-STARTPTS,fps=24,tpad=stop_mode=clone:stop_duration=0.25",
                 "-af", "asetpts=PTS-STARTPTS,apad", "-t", str(target),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                 "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(output)])
    return validate_av(output, target)


HTML = '''<!doctype html><meta charset="utf-8"><title>AI-generated H3 avatar</title>
<h1>AI-generated avatar — sequential buffered segments</h1>
<button id="start">Start / resume playback</button><p id="status">Waiting</p>
<video id="video" controls playsinline width="720"></video>
<p><a href="demo.mp4">Saved complete demo (available after completion)</a></p>
<script>
const v=document.querySelector('#video'), label=document.querySelector('#status');
let queue=[], next=0, playing=false, enabled=false;
function advance(){
 if(!enabled||playing||next>=queue.length)return;
 v.src=queue[next++].file;playing=true;
 v.play().catch(()=>{playing=false;next--;enabled=false;label.textContent='Click Start to play';});
}
v.onended=()=>{playing=false;advance();};
document.querySelector('#start').onclick=()=>{enabled=true;advance();};
async function poll(){try{
 const s=await (await fetch('status.json',{cache:'no-store'})).json();
 queue=s.segments;label.textContent=s.state+' — '+queue.length+'/3 ready';advance();
}catch(e){label.textContent='Waiting for status';}setTimeout(poll,1000);}poll();
</script>'''


def save_status(root, status):
    tmp = root / "status.tmp"
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(root / "status.json")


def orchestrate(args, root, status):
    try:
        image = Path(args.reference_image)
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}.get(image.suffix.lower())
        if not mime:
            raise ValueError("Reference image must be PNG/JPEG/WEBP")
        image_uri = data_uri(image, mime, 30 * 1024**2)
        tts = LocalVoxCPM2(args.tts_model, Path(args.reference_audio))
        history = []
        for turn in range(3):
            status["state"] = f"generating turn {turn + 1}"
            save_status(root, status)
            text = agent_utterance(args, history, turn)
            original = root / f"tts-{turn}.wav"
            tts.synthesize(text, original)
            audios = [s for s in probe(original)["streams"] if s["codec_type"] == "audio"]
            if len(audios) != 1:
                raise ValueError("TTS must return one audio stream")
            spoken = stream_duration(audios[0])
            duration = duration_for_tts(spoken)
            padded = root / f"driving-{turn}.wav"
            run_command(["ffmpeg", "-y", "-v", "error", "-i", str(original), "-af", "apad",
                         "-t", str(duration), "-ar", "32000", "-ac", "2", "-c:a", "pcm_s16le", str(padded)])
            raw = root / f"h3-{turn}.mp4"
            timing = generate_h3(args, image_uri, padded, duration, text, raw, turn)
            segment = root / f"segment-{turn}.mp4"
            av = mux_original(raw, original, segment, duration, spoken)
            status["segments"].append({"file": segment.name, "text": text, "duration": duration,
                                       "tts_seconds": spoken, "av_seconds": av, "generation": timing})
            save_status(root, status)  # Publish only after decode + A/V validation.
        listing = root / "concat.txt"
        listing.write_text("".join(f"file 'segment-{i}.mp4'\n" for i in range(3)))
        demo = root / "demo.mp4"
        run_command(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "1", "-i", str(listing),
                     "-c:v", "libx264", "-c:a", "aac", "-movflags", "+faststart", str(demo)])
        validate_av(demo, sum(s["duration"] for s in status["segments"]), tolerance=0.3)
        status.update(state="complete", demo="demo.mp4")
    except Exception as exc:
        status.update(state="failed", error=type(exc).__name__)
        # Do not put HTTP errors (potentially echoed personal payloads) in status.
        raise
    finally:
        save_status(root, status)


class DemoHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        name = urlsplit(self.path).path
        if name not in ("/", "/index.html", "/status.json", "/demo.mp4",
                        "/segment-0.mp4", "/segment-1.mp4", "/segment-2.mp4"):
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if urlsplit(self.path).path not in ("/", "/index.html", "/status.json", "/demo.mp4",
                                            "/segment-0.mp4", "/segment-1.mp4", "/segment-2.mp4"):
            self.send_error(404)
            return
        super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm-url", default="http://127.0.0.1:8000/v1/chat/completions")
    p.add_argument("--llm-model", required=True)
    p.add_argument("--h3-url", default="http://127.0.0.1:8091/v1/videos/sync")
    p.add_argument("--h3-model", default="MiniMaxAI/MiniMax-H3")
    p.add_argument("--tts-model", default="openbmb/VoxCPM2", help="VoxCPM2 ID or downloaded local directory")
    p.add_argument("--reference-image", required=True)
    p.add_argument("--reference-audio", required=True)
    p.add_argument("--agent-instruction", default="Speak in Japanese.")
    p.add_argument("--output", required=True, help="New private output directory")
    p.add_argument("--allow-remote-personal-data", action="store_true",
                   help="Explicit consent to send generated speech/text and personal image to remote endpoints")
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=1344)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--timeout", type=float, default=14400)
    args = p.parse_args(argv)
    if min(args.width, args.height) <= 0 or args.width % 32 or args.height % 32:
        p.error("Canvas must have positive multiples of 32")
    for url in (args.llm_url, args.h3_url):
        check_endpoint(url, args.allow_remote_personal_data)
    for path in (args.reference_image, args.reference_audio):
        if not Path(path).is_file():
            p.error(f"Missing reference: {path}")
    root = Path(args.output).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    (root / "index.html").write_text(HTML, encoding="utf-8")
    status = {"state": "starting", "segments": [], "delivery": "sequential buffered segments, not causal streaming",
              "personal_lora": "not requested; server adapters unverified", "ai_generated": True}
    save_status(root, status)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), functools.partial(DemoHandler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"Browser: http://127.0.0.1:{server.server_port}/ (click Start); Ctrl-C to stop", flush=True)
    try:
        orchestrate(args, root, status)
        print(f"Saved: {root / 'demo.mp4'}; serving until Ctrl-C", flush=True)
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        status["state"] = "stopped" if status["state"] != "complete" else "complete"
        save_status(root, status)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
