"""Small read-only obs-websocket 5 / RPC 1 core (no network implementation).

Official contract: https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md

Injected transport: send(text), recv(timeout=<remaining seconds>) -> JSON text,
close(). recv MUST enforce that timeout and raise TimeoutError on expiry; send
and close must also be bounded by the adapter. No threads can safely enforce a
blocking foreign transport's deadline. Adapters must not log frames/credentials.

Use ``with Session(transport, source_name, ...) as session:`` then
``session.request("GetVersion")`` etc. request returns ONLY sanitized summaries,
never raw responseData. Alternatively collect_audit owns the entire session and
returns partial observations plus fixed error codes on failure. No CLI, socket,
device discovery, settings changes, replay or reconnect is provided.
"""
import base64
import hashlib
import json
import math
import re
import time

INPUT_VOLUME_METERS = 1 << 16
ALLOWED_REQUESTS = frozenset({
    "GetVersion", "GetVideoSettings", "GetVirtualCamStatus", "GetInputList",
    "GetInputMute", "GetInputVolume", "GetInputAudioMonitorType", "GetInputSettings",
})
SOURCE_REQUESTS = ALLOWED_REQUESTS - {
    "GetVersion", "GetVideoSettings", "GetVirtualCamStatus", "GetInputList"}
MONITOR_TYPES = frozenset({"OBS_MONITORING_TYPE_NONE", "OBS_MONITORING_TYPE_MONITOR_ONLY",
                           "OBS_MONITORING_TYPE_MONITOR_AND_OUTPUT"})


class ProtocolError(Exception):
    """Fixed local error code only; never server comments or transport messages."""


def authentication_response(password, salt, challenge):
    """OBS: Base64(SHA256(Base64(SHA256(password + salt)) + challenge))."""
    if not all(isinstance(x, str) for x in (password, salt, challenge)):
        raise ProtocolError("authentication_invalid")
    try:
        secret = base64.b64encode(hashlib.sha256((password + salt).encode("utf-8")).digest())
        return base64.b64encode(hashlib.sha256(secret + challenge.encode("utf-8")).digest()).decode("ascii")
    except UnicodeError:
        raise ProtocolError("authentication_invalid") from None


def _number(value, *, nonnegative=False):
    try:
        return (type(value) in (int, float) and math.isfinite(value)
                and (not nonnegative or value >= 0))
    except OverflowError:
        return False


def _require(condition):
    if not condition:
        raise ProtocolError("response_invalid")


class Session:
    """Single-use, sequential session; no raw request/opcode escape hatch.

    Deadline includes handshake/queries/meter collection. Defaults: 10 seconds,
    256 received messages, 100 matching meter events, 256 KiB/text message.
    Hard ceilings: 30 seconds, 1000 messages, 400 meter events. Request IDs are
    monotonic session-local strings, never caller-supplied or reused.
    Explicit meters=True subscribes ONLY to InputVolumeMeters; otherwise 0.
    """

    def __init__(self, transport, source_name, *, password=None, meters=False,
                 timeout=10.0, max_messages=256, max_meter_events=100,
                 clock=time.monotonic):
        if (not isinstance(source_name, str) or not source_name or len(source_name) > 512
                or type(meters) is not bool or not _number(timeout) or not 0 < timeout <= 30
                or type(max_messages) is not int or not 2 <= max_messages <= 1000
                or type(max_meter_events) is not int or not 1 <= max_meter_events <= 400
                or (password is not None and not isinstance(password, str))):
            raise ProtocolError("configuration_invalid")
        self._transport, self._source = transport, source_name
        self._password = password
        self._meters, self._clock, self._timeout = meters, clock, timeout
        self._max_messages, self._max_events = max_messages, max_meter_events
        self._deadline = None
        self._messages = self._sequence = 0
        self._entered = self._ready = self._closed = False
        self._selected = False
        self._uuid = None
        self._meter_count = 0
        self._peak = self._rms = None

    def __enter__(self):
        if self._entered or self._closed:
            raise ProtocolError("session_reused")
        self._entered = True
        self._deadline = self._clock() + self._timeout
        try:
            self._handshake()
            return self
        except Exception:
            self.close()
            raise

    def __exit__(self, *_):
        self.close()

    def close(self):
        if not self._closed:
            self._closed, self._ready = True, False
            self._password = None
            try:
                self._transport.close()
            except Exception:
                pass  # Never expose a transport exception containing credentials.

    def _remaining(self):
        remaining = self._deadline - self._clock()
        if remaining <= 0:
            raise ProtocolError("deadline_exceeded")
        return remaining

    def _send(self, op, data):
        self._remaining()
        try:
            self._transport.send(json.dumps({"op": op, "d": data}, allow_nan=False))
        except Exception:
            raise ProtocolError("transport_send_failed") from None
        self._remaining()

    def _recv(self):
        if self._messages >= self._max_messages:
            raise ProtocolError("message_limit")
        remaining = self._remaining()
        try:
            text = self._transport.recv(timeout=remaining)
        except TimeoutError:
            raise ProtocolError("read_timeout") from None
        except Exception:
            raise ProtocolError("transport_receive_failed") from None
        self._messages += 1
        self._remaining()
        if not isinstance(text, str) or len(text) > 262144:
            raise ProtocolError("message_invalid")
        try:
            message = json.loads(text)
        except (ValueError, RecursionError):
            raise ProtocolError("message_invalid") from None
        if (not isinstance(message, dict) or type(message.get("op")) is not int
                or not isinstance(message.get("d"), dict)):
            raise ProtocolError("message_invalid")
        return message["op"], message["d"]

    def _handshake(self):
        op, hello = self._recv()
        if op != 0 or type(hello.get("rpcVersion")) is not int or hello["rpcVersion"] < 1:
            raise ProtocolError("hello_invalid")
        identify = {"rpcVersion": 1,
                    "eventSubscriptions": INPUT_VOLUME_METERS if self._meters else 0}
        if "authentication" in hello:
            auth = hello["authentication"]
            if (not isinstance(auth, dict) or self._password is None
                    or not all(isinstance(auth.get(k), str) and 0 < len(auth[k]) <= 4096
                               for k in ("salt", "challenge"))):
                raise ProtocolError("authentication_required_or_invalid")
            identify["authentication"] = authentication_response(
                self._password, auth["salt"], auth["challenge"])
        self._password = None
        self._send(1, identify)
        op, identified = self._recv()
        if op != 2 or type(identified.get("negotiatedRpcVersion")) is not int or identified["negotiatedRpcVersion"] != 1:
            raise ProtocolError("identified_invalid")
        self._ready = True

    def request(self, request_type, request_data=None):
        """Allowlisted getter; source getters accept only the selected inputName.

        Call GetInputList first to establish exactly one ffmpeg_source selection.
        Wrong request IDs/types (including stale duplicates) fail closed, while
        events are consumed/aggregated without disrupting the pending response.
        OBS success is result=true AND code=100 (not code=0).
        """
        if not isinstance(request_type, str) or request_type not in ALLOWED_REQUESTS:
            raise ProtocolError("request_disallowed")
        expected = {"inputName": self._source} if request_type in SOURCE_REQUESTS else {}
        if request_data is not None and (type(request_data) is not dict or request_data != expected):
            raise ProtocolError("request_data_disallowed")
        if not self._ready or self._closed:
            raise ProtocolError("session_not_ready")
        if request_type in SOURCE_REQUESTS and not self._selected:
            raise ProtocolError("source_not_selected_ffmpeg")
        self._sequence += 1
        request_id = str(self._sequence)
        self._send(6, {"requestType": request_type, "requestId": request_id,
                       "requestData": expected})
        while True:
            op, data = self._recv()
            if op == 5:
                self._event(data)
                continue
            if (op != 7 or data.get("requestId") != request_id
                    or data.get("requestType") != request_type):
                raise ProtocolError("response_mismatch")
            status = data.get("requestStatus")
            if (not isinstance(status, dict) or status.get("result") is not True
                    or type(status.get("code")) is not int or status["code"] != 100):
                raise ProtocolError("request_failed")
            response = data.get("responseData", {})
            _require(isinstance(response, dict))
            return self._summary(request_type, response)

    def _summary(self, kind, data):
        if kind == "GetVersion":
            result = {}
            for key in ("obsVersion", "obsWebSocketVersion"):
                value = data.get(key)
                _require(isinstance(value, str) and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][a-zA-Z0-9.-]+)?", value) is not None
                         and len(value) <= 64)
                result[key] = value
            return result
        if kind == "GetVideoSettings":
            keys = ("baseWidth", "baseHeight", "outputWidth", "outputHeight", "fpsNumerator", "fpsDenominator")
            _require(all(type(data.get(k)) is int and 0 < data[k] <= 1000000 for k in keys))
            return {k: data[k] for k in keys}
        if kind == "GetVirtualCamStatus":
            _require(type(data.get("outputActive")) is bool)
            return {"outputActive": data["outputActive"]}
        if kind == "GetInputList":
            self._selected, self._uuid = False, None
            inputs = data.get("inputs")
            _require(isinstance(inputs, list) and len(inputs) <= 1024
                     and all(isinstance(i, dict) for i in inputs))
            matches = [i for i in inputs if i.get("inputName") == self._source]
            self._selected = len(matches) == 1 and matches[0].get("inputKind") == "ffmpeg_source"
            if self._selected:
                uuid = matches[0].get("inputUuid")
                _require(uuid is None or (isinstance(uuid, str) and 0 < len(uuid) <= 128))
                self._uuid = uuid
            return {"input_count": len(inputs), "selected_matches": len(matches),
                    "selected_ffmpeg_source": self._selected}
        if kind == "GetInputMute":
            _require(type(data.get("inputMuted")) is bool)
            return {"inputMuted": data["inputMuted"]}
        if kind == "GetInputVolume":
            _require(_number(data.get("inputVolumeMul"), nonnegative=True)
                     and _number(data.get("inputVolumeDb")))
            return {k: data[k] for k in ("inputVolumeMul", "inputVolumeDb")}
        if kind == "GetInputAudioMonitorType":
            _require(isinstance(data.get("monitorType"), str) and data["monitorType"] in MONITOR_TYPES)
            return {"monitorType": data["monitorType"]}
        _require(data.get("inputKind") == "ffmpeg_source" and isinstance(data.get("inputSettings"), dict))
        settings = data["inputSettings"]
        # Never return strings, URLs, paths, credentials or the settings object.
        return {"ffmpeg_source": True, **{k: settings[k] for k in
                ("is_local_file", "looping", "restart_on_activate", "close_when_inactive")
                if type(settings.get(k)) is bool}}

    def _event(self, data):
        if not self._meters or not self._selected or data.get("eventType") != "InputVolumeMeters":
            return
        event = data.get("eventData")
        _require(isinstance(event, dict) and isinstance(event.get("inputs"), list)
                 and len(event["inputs"]) <= 1024)
        matches = [i for i in event["inputs"] if isinstance(i, dict) and
                   (i.get("inputUuid") == self._uuid if self._uuid is not None
                    else i.get("inputName") == self._source)]
        if not matches:
            return
        _require(len(matches) == 1)
        if self._meter_count >= self._max_events:
            raise ProtocolError("meter_event_limit")
        levels = matches[0].get("inputLevelsMul")
        _require(isinstance(levels, list) and 1 <= len(levels) <= 8)
        # OBS channel tuple: magnitude (RMS), peak, input peak; linear amplitudes.
        _require(all(isinstance(ch, list) and len(ch) == 3 and
                     all(_number(v, nonnegative=True) and v <= 1e6 for v in ch) for ch in levels))
        rms, peak = max(ch[0] for ch in levels), max(ch[1] for ch in levels)
        self._rms = rms if self._rms is None else max(self._rms, rms)
        self._peak = peak if self._peak is None else max(self._peak, peak)
        self._meter_count += 1

    def meter_summary(self):
        """Max OBS-reported per-channel RMS/peak, NOT PCM/time-integrated RMS."""
        return {"enabled": self._meters, "number_events": self._meter_count,
                "max_peak_mul": self._peak, "max_rms_mul": self._rms,
                "runtime_verified": self._meter_count > 0,
                "blackhole_signal_verified": False, "remote_verified": False}

    def collect_meters(self):
        """Consume events until event cap or original deadline; never renew it."""
        if not self._ready or not self._meters or not self._selected:
            raise ProtocolError("meter_collection_disallowed")
        while self._meter_count < self._max_events:
            try:
                op, data = self._recv()
            except ProtocolError as error:
                if str(error) in ("read_timeout", "deadline_exceeded"):
                    break
                raise
            if op != 5:
                raise ProtocolError("unexpected_message")
            self._event(data)
        return self.meter_summary()


def collect_audit(transport, source_name, **options):
    """Own/close transport; return safe per-request observations and meter summary.

    Top-level runtime_verified stays false: individual OBS states alone are
    verified. Monitoring-device selection is NOT exposed by these protocol APIs;
    GetInputSettings is source configuration, not OBS Settings -> Audio.
    """
    report = {"read_only": True, "runtime_verified": False,
              "verification_scope": "individual_obs_states_only",
              "remote_verified": False, "blackhole_signal_verified": False,
              "monitoring_device_verified": False, "observations": {}, "errors": []}
    session = None
    try:
        session = Session(transport, source_name, **options)
        with session:
            for kind in ("GetVersion", "GetVideoSettings", "GetVirtualCamStatus", "GetInputList",
                         "GetInputMute", "GetInputVolume", "GetInputAudioMonitorType", "GetInputSettings"):
                summary = session.request(kind)
                report["observations"][kind] = {"runtime_verified": True, "summary": summary}
            if session._meters:
                session.collect_meters()
    except ProtocolError as error:
        report["errors"].append(str(error))
    finally:
        if session is not None:
            session.close()
            report["meters"] = session.meter_summary()
        else:
            try:
                transport.close()
            except Exception:
                pass
    return report
