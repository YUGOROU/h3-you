"""Protocol fixtures only; no sockets, installed OBS, devices or settings."""
import base64
import hashlib
import json
import unittest

from research.h3_personal.obs_runtime import (
    ALLOWED_REQUESTS, INPUT_VOLUME_METERS, ProtocolError, Session,
    authentication_response, collect_audit,
)

SOURCE = "selected generated source"
SECRET = "private-password-never-log"
UUID = "selected-uuid"


def frame(op, **data):
    return json.dumps({"op": op, "d": data})


def event(levels=None, uuid=UUID):
    return frame(5, eventType="InputVolumeMeters", eventIntent=INPUT_VOLUME_METERS,
                 eventData={"inputs": [{"inputName": SOURCE, "inputUuid": uuid,
                                         "inputLevelsMul": levels or [[0.1, 0.5, 0.7]]}]})


RESPONSES = {
    "GetVersion": {"obsVersion": "32.0.1", "obsWebSocketVersion": "5.6.3",
                   "availableRequests": ["SetInputSettings"], "secret": SECRET},
    "GetVideoSettings": dict(baseWidth=1920, baseHeight=1080, outputWidth=1280,
                             outputHeight=720, fpsNumerator=30, fpsDenominator=1),
    "GetVirtualCamStatus": {"outputActive": True},
    "GetInputList": {"inputs": [{"inputName": SOURCE, "inputUuid": UUID,
                                 "inputKind": "ffmpeg_source", "password": SECRET},
                                {"inputName": SECRET, "inputKind": SECRET}]},
    "GetInputMute": {"inputMuted": False},
    "GetInputVolume": {"inputVolumeMul": 1.0, "inputVolumeDb": 0.0},
    "GetInputAudioMonitorType": {"monitorType": "OBS_MONITORING_TYPE_MONITOR_ONLY"},
    "GetInputSettings": {"inputKind": "ffmpeg_source", "inputSettings": {
        "input": "udp://user:private-password-never-log@example.invalid:1234",
        "password": SECRET, "local_file": SECRET, "is_local_file": False,
        "looping": True, "restart_on_activate": "not-a-boolean"}},
}


class Transport:
    def __init__(self, *, auth=False, hook=None):
        hello = {"rpcVersion": 1}
        if auth:
            hello["authentication"] = {"salt": "salt", "challenge": "challenge"}
        self.queue = [frame(0, **hello), frame(2, negotiatedRpcVersion=1)]
        self.sent, self.timeouts = [], []
        self.closed = 0
        self.hook = hook

    def send(self, text):
        self.sent.append(json.loads(text))
        message = self.sent[-1]
        if message["op"] == 6:
            data = message["d"]
            response = dict(requestId=data["requestId"], requestType=data["requestType"],
                            requestStatus={"result": True, "code": 100},
                            responseData=RESPONSES[data["requestType"]])
            before = []
            if self.hook:
                before = self.hook(data, response) or []
            self.queue.extend(before + [frame(7, **response)])

    def recv(self, *, timeout):
        self.timeouts.append(timeout)
        if not self.queue:
            raise TimeoutError(SECRET)
        return self.queue.pop(0)

    def close(self):
        self.closed += 1


class OBSRuntimeTests(unittest.TestCase):
    def test_authentication_exact_algorithm_and_no_plaintext(self):
        secret = base64.b64encode(hashlib.sha256((SECRET + "salt").encode()).digest()).decode()
        expected = base64.b64encode(hashlib.sha256((secret + "challenge").encode()).digest()).decode()
        self.assertEqual(authentication_response(SECRET, "salt", "challenge"), expected)
        transport = Transport(auth=True)
        report = collect_audit(transport, SOURCE, password=SECRET)
        self.assertEqual(transport.sent[0], {"op": 1, "d": {
            "rpcVersion": 1, "eventSubscriptions": 0, "authentication": expected}})
        self.assertNotIn(SECRET, json.dumps(transport.sent))
        self.assertNotIn(SECRET, json.dumps(report))
        self.assertEqual(transport.closed, 1)

    def test_complete_report_is_individual_obs_state_only(self):
        transport = Transport()
        report = collect_audit(transport, SOURCE)
        self.assertEqual(report["errors"], [])
        self.assertEqual(set(report["observations"]), ALLOWED_REQUESTS)
        self.assertTrue(all(o["runtime_verified"] for o in report["observations"].values()))
        for key in ("runtime_verified", "remote_verified", "blackhole_signal_verified",
                    "monitoring_device_verified"):
            self.assertFalse(report[key])
        self.assertEqual(report["observations"]["GetInputList"]["summary"], {
            "input_count": 2, "selected_matches": 1, "selected_ffmpeg_source": True})
        text = json.dumps(report, allow_nan=False)
        for forbidden in (SECRET, "udp://", "inputSettings", UUID):
            self.assertNotIn(forbidden, text)
        self.assertFalse(report["meters"]["runtime_verified"])
        self.assertTrue(all(0 < t <= 10 for t in transport.timeouts))

    def test_mutations_unknown_getters_and_raw_data_disallowed(self):
        transport = Transport()
        with Session(transport, SOURCE) as session:
            for kind in ("SetInputMute", "StartVirtualCam", "StopVirtualCam", "ToggleInputMute",
                         "GetSettings", "GetSourceScreenshot", {"op": 6}, "CallVendorRequest"):
                with self.subTest(kind=kind), self.assertRaisesRegex(ProtocolError, "request_disallowed"):
                    session.request(kind)
            for data in ({"inputName": "other"}, {"inputUuid": UUID}, {"inputName": SOURCE, "set": True}):
                with self.assertRaisesRegex(ProtocolError, "request_data_disallowed"):
                    session.request("GetInputSettings", data)
            with self.assertRaisesRegex(ProtocolError, "source_not_selected_ffmpeg"):
                session.request("GetInputSettings")
        self.assertEqual(len(transport.sent), 1)

    def test_wrong_request_id_type_and_stale_id_fail_closed(self):
        for field, value in (("requestId", "wrong-uuid"), ("requestType", "SetInputMute")):
            def hook(_, response):
                response[field] = value
            transport = Transport(hook=hook)
            report = collect_audit(transport, SOURCE)
            self.assertEqual(report["errors"], ["response_mismatch"])
            self.assertEqual(transport.closed, 1)
        transport = Transport()
        with Session(transport, SOURCE) as session:
            session.request("GetVersion")
            transport.queue.append(frame(7, requestId="1", requestType="GetVersion"))
            with self.assertRaisesRegex(ProtocolError, "response_mismatch"):
                session.request("GetVersion")
        self.assertEqual([m["d"]["requestId"] for m in transport.sent[1:]], ["1", "2"])

    def test_status_100_only_and_server_comment_not_exposed(self):
        for result, code in ((False, 100), (True, 0), (False, 600), (True, 101)):
            def hook(_, response):
                response["requestStatus"] = {"result": result, "code": code, "comment": SECRET}
            report = collect_audit(Transport(hook=hook), SOURCE)
            self.assertEqual(report["errors"], ["request_failed"])
            self.assertNotIn(SECRET, json.dumps(report))

    def test_interleaved_meters_uuid_filter_and_aggregate(self):
        def hook(request, _):
            if request["requestType"] == "GetInputMute":
                return [event(uuid="other"), event(), event([[0.3, 0.8, 0.9], [0.2, 0.4, 0.5]])]
        transport = Transport(hook=hook)
        report = collect_audit(transport, SOURCE, meters=True, max_meter_events=2)
        self.assertEqual(report["errors"], [])
        self.assertEqual(transport.sent[0]["d"]["eventSubscriptions"], 1 << 16)
        meters = report["meters"]
        self.assertEqual(meters["number_events"], 2)
        self.assertEqual(meters["max_peak_mul"], 0.8)
        self.assertEqual(meters["max_rms_mul"], 0.3)
        self.assertTrue(meters["runtime_verified"])
        self.assertFalse(meters["blackhole_signal_verified"])
        self.assertNotIn("inputLevelsMul", json.dumps(report))

    def test_meter_opt_in_and_no_events_not_signal(self):
        def hook(request, _):
            return [event()] if request["requestType"] == "GetInputMute" else []
        report = collect_audit(Transport(hook=hook), SOURCE)
        self.assertEqual(report["meters"]["number_events"], 0)
        report = collect_audit(Transport(), SOURCE, meters=True)
        self.assertEqual(report["errors"], [])  # bounded recv timeout ends meter window
        self.assertFalse(report["meters"]["runtime_verified"])

    def test_meter_limit_and_invalid_levels(self):
        def flood(request, _):
            return [event(), event()] if request["requestType"] == "GetInputMute" else []
        report = collect_audit(Transport(hook=flood), SOURCE, meters=True, max_meter_events=1)
        self.assertEqual(report["errors"], ["meter_event_limit"])
        self.assertEqual(report["meters"]["number_events"], 1)
        for invalid in (float("nan"), float("inf"), -1, True, "secret", 10 ** 400):
            def bad(request, _):
                return [event([[invalid, 0.2, 0.3]])] if request["requestType"] == "GetInputMute" else []
            report = collect_audit(Transport(hook=bad), SOURCE, meters=True)
            self.assertEqual(report["errors"], ["response_invalid"])
            self.assertEqual(report["meters"]["number_events"], 0)

    def test_message_and_deadline_bounds_and_finally_close(self):
        transport = Transport(hook=lambda *_: [frame(5, eventType="ignored")] * 5)
        report = collect_audit(transport, SOURCE, max_messages=3)
        self.assertEqual(report["errors"], ["message_limit"])
        self.assertEqual(transport.closed, 1)
        now = [0.0]
        transport = Transport()
        original = transport.recv
        def recv(**kwargs):
            result = original(**kwargs)
            now[0] += 6
            return result
        transport.recv = recv
        report = collect_audit(transport, SOURCE, clock=lambda: now[0])
        self.assertEqual(report["errors"], ["deadline_exceeded"])
        self.assertEqual(transport.closed, 1)

    def test_handshake_errors_transport_failure_and_session_reuse(self):
        for hello in (frame(0, rpcVersion=0), frame(7), "invalid " + SECRET):
            transport = Transport()
            transport.queue[0] = hello
            report = collect_audit(transport, SOURCE)
            self.assertTrue(report["errors"])
            self.assertNotIn(SECRET, json.dumps(report))
            self.assertEqual(transport.closed, 1)
        transport = Transport(auth=True)
        report = collect_audit(transport, SOURCE)
        self.assertEqual(report["errors"], ["authentication_required_or_invalid"])
        transport = Transport()
        def recv(**_):
            raise OSError(SECRET)
        transport.recv = recv
        report = collect_audit(transport, SOURCE)
        self.assertEqual(report["errors"], ["transport_receive_failed"])
        session = Session(Transport(), SOURCE)
        with session:
            pass
        with self.assertRaisesRegex(ProtocolError, "session_reused"):
            session.__enter__()

    def test_non_ffmpeg_or_duplicate_source_never_gets_settings(self):
        for inputs in ([{"inputName": SOURCE, "inputKind": "camera"}],
                       [{"inputName": SOURCE, "inputKind": "ffmpeg_source"}] * 2):
            def hook(request, response):
                if request["requestType"] == "GetInputList":
                    response["responseData"] = {"inputs": inputs}
            transport = Transport(hook=hook)
            report = collect_audit(transport, SOURCE)
            self.assertEqual(report["errors"], ["source_not_selected_ffmpeg"])
            self.assertNotIn("GetInputSettings", [m["d"].get("requestType") for m in transport.sent])

    def test_malformed_binary_oversized_and_send_failure(self):
        for payload in (b"binary", "x" * 262145, frame(True), frame(0, rpcVersion=True)):
            transport = Transport()
            transport.queue[0] = payload
            report = collect_audit(transport, SOURCE)
            self.assertTrue(report["errors"])
            self.assertEqual(transport.closed, 1)
        transport = Transport()
        def send(_):
            raise OSError(SECRET)
        transport.send = send
        report = collect_audit(transport, SOURCE, password=SECRET)
        self.assertEqual(report["errors"], ["transport_send_failed"])
        self.assertNotIn(SECRET, json.dumps(report))
        self.assertEqual(transport.closed, 1)

    def test_invalid_configuration_closes_owned_transport(self):
        for options in ({"timeout": float("inf")}, {"timeout": 31}, {"max_messages": 1001},
                        {"max_meter_events": 401}, {"meters": 1}):
            transport = Transport()
            report = collect_audit(transport, SOURCE, **options)
            self.assertEqual(report["errors"], ["configuration_invalid"])
            self.assertEqual(transport.closed, 1)


if __name__ == "__main__":
    unittest.main()
