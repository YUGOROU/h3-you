"""Offline OBS saved-configuration audit; never applies settings or probes devices.

``audit`` accepts INI text and decoded scene JSON without mutating either.
``audit_files`` reads only the two explicitly supplied files. No discovery, logs,
websocket access, media probing, processes, or runtime verification is performed.
Expected source dimensions describe the supplied contract, NOT measured media.
"""
import argparse
import configparser
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

from research.prepare_h3_obs_generated import AUDIO_KEYS

MONITORING_MODES = {0: "none", 1: "monitor_only", 2: "monitor_and_output"}
EXPECTED_MONITOR_DEVICE = "BlackHole2ch_UID"


def _duplicate_monitor_keys(text):
    """Inspect explicit keys across all sections, without selecting a winner.

    Ignore comments and indented continuation values, just as ConfigParser does.
    Values and section names are never included in diagnostics.
    """
    counts = {}
    if not isinstance(text, str):
        return []
    section = False
    option_indent = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        indent = len(line) - len(line.lstrip())
        if option_indent is not None and indent > option_indent:
            continue
        if stripped.startswith("[") and "]" in stripped:
            section = True
            option_indent = None
            continue
        match = re.match(r"([^:=]+?)\s*[:=]", stripped)
        if section and match:
            option_indent = indent
            key = match.group(1).strip().lower()
            if key in ("monitoringdeviceid", "monitoringdevicename"):
                counts[key] = counts.get(key, 0) + 1
    return sorted(key for key, count in counts.items() if count > 1)


def audit(profile_ini, scene_data, source_name, *, expected_source_size=(252, 448)):
    """Return allowlisted findings only (never arbitrary source settings/URLs).

    Malformed inputs produce fixed diagnostic messages, not parser exceptions
    containing potentially private configuration. Monitoring enum values follow
    OBS: 0=none, 1=monitor only, 2=monitor and output. Both 1 and 2 can feed the
    monitoring device; mixer output is a separate path from virtual-camera video.
    """
    result = {
        "recorded_config_only": True, "runtime_verified": False,
        "remote_verified": False, "read_only": True,
        "findings": [],
        "source_contract": {"width": None, "height": None, "measured": False},
    }

    def finding(code, severity, message):
        result["findings"].append(dict(code=code, severity=severity, message=message))

    width, height = expected_source_size
    if (type(width) is not int or type(height) is not int or width <= 0 or height <= 0):
        raise ValueError("expected_source_size must contain two positive integers")
    result["source_contract"].update(width=width, height=height)
    finding("verification_scope", "info",
            "Saved configuration only; no audio signal, active scene, browser hook, device, or remote delivery verified.")
    finding("source_aspect", "info" if width < height else "warning",
            "Expected source is portrait; this is an input contract, not measured media."
            if width < height else "Expected source contract is not portrait.")
    duplicates = _duplicate_monitor_keys(profile_ini)
    if duplicates:
        finding("profile_monitor_conflict", "error",
                "Duplicate critical monitoring keys make the saved route ambiguous; no route selected.")
    profile = configparser.ConfigParser(interpolation=None)
    try:
        profile.read_string(profile_ini)
    except (configparser.Error, TypeError):
        finding("profile_invalid", "error", "Profile INI could not be parsed.")
        profile = configparser.ConfigParser(interpolation=None)
    # Do not inherit INI DEFAULT entries: the monitor selection must be explicit.
    audio = profile._sections.get("Audio", {})
    device = audio.get("monitoringdeviceid")
    if duplicates:
        finding("monitor_device", "error", "Saved monitoring route is ambiguous due to duplicate critical keys; no route selected.")
    elif device == EXPECTED_MONITOR_DEVICE:
        finding("monitor_device", "info", "Explicit BlackHole2ch_UID monitoring device is saved; runtime route is unverified.")
    elif device is None or not device.strip():
        finding("monitor_device", "error", "Explicit monitoring device is missing.")
    elif device.strip().lower() in ("default", "devicedefault"):
        finding("monitor_device", "error", "Default monitoring device is not an explicit BlackHole route.")
    else:
        finding("monitor_device", "error", "Monitoring device differs from the required explicit BlackHole route.")

    for prefix, code in (("Base", "canvas"), ("Output", "output")):
        video = profile._sections.get("Video", {})
        try:
            w, h = int(video[prefix.lower() + "cx"]), int(video[prefix.lower() + "cy"])
            if w <= 0 or h <= 0:
                raise ValueError
        except (KeyError, ValueError, TypeError):
            finding(code + "_geometry", "warning", "Saved video geometry is missing or invalid.")
            continue
        if abs(w / h - 16 / 9) < 0.002:
            finding(code + "_geometry", "info",
                    "Saved 16:9 transport can contain centered full-height portrait with sidebars; sidebars do not imply a landscape model source. Browser extraction requires prepared black sidebars and a live hook, neither verified here.")
        elif abs(w / h - width / height) < 0.002:
            finding(code + "_geometry", "info",
                    "Saved geometry matches expected portrait; virtual-camera negotiation and remote layout remain unverified. The portrait JS contract requires 16:9 input and otherwise retains the original stream.")
        else:
            finding(code + "_geometry", "warning", "Saved geometry matches neither the source contract nor 16:9 transport; inspect layout without assuming model aspect failure.")

    if not isinstance(scene_data, dict):
        finding("scene_invalid", "error", "Scene collection must be an object.")
        return result
    # prepare_h3_obs_generated.collection intentionally uses absent/null globals.
    enabled = [key for key in AUDIO_KEYS if scene_data.get(key) is not None]
    finding("global_audio_capture", "error" if enabled else "info",
            "Global audio capture objects are configured; no-auto-capture policy fails."
            if enabled else "Global audio capture objects are absent/null in this saved collection.")
    sources = scene_data.get("sources")
    if not isinstance(sources, list) or any(not isinstance(s, dict) for s in sources):
        finding("sources_invalid", "error", "Saved source list is missing or invalid.")
        return result
    capture_types = {"coreaudio_input_capture", "coreaudio_output_capture", "wasapi_input_capture",
                     "wasapi_output_capture", "pulse_input_capture", "pulse_output_capture",
                     "alsa_input_capture", "sck_audio_capture"}
    if any(s.get("id") in capture_types or s.get("versioned_id") in capture_types for s in sources):
        finding("explicit_audio_capture", "error", "An explicit audio capture source is saved; inspect sensor policy separately.")
    matches = [s for s in sources if s.get("name") == source_name]
    if len(matches) != 1:
        finding("generated_source", "error", "Generated source name is missing or ambiguous.")
        return result
    source = matches[0]
    if source.get("id") != "ffmpeg_source":
        finding("generated_source", "error", "Selected source is not an OBS ffmpeg media source.")
    settings = source.get("settings")
    settings = settings if isinstance(settings, dict) else {}
    try:
        udp = (settings.get("is_local_file") is False
               and urlsplit(settings.get("input", "")).scheme.lower() == "udp")
    except (TypeError, ValueError, AttributeError):
        udp = False
    finding("generated_udp", "info" if udp else "error",
            "Selected source records UDP media input; signal and audio tracks are unverified."
            if udp else "Selected source does not record an explicit non-local UDP media input.")
    mode = source.get("monitoring_type")
    if type(mode) is not int or mode not in MONITORING_MODES:
        finding("source_monitoring", "error", "Monitoring mode is missing or invalid.")
    else:
        result["monitoring_mode"] = MONITORING_MODES[mode]
        finding("source_monitoring", "error" if mode == 0 else "info",
                ("Monitoring is off (0); no monitor-device audio path."
                 if mode == 0 else "Monitor only (1): feeds monitoring device, not program output."
                 if mode == 1 else "Monitor and output (2): feeds monitoring device and program output; inspect duplicate routing."))
    muted = source.get("muted")
    finding("source_mute", "info" if muted is False else "error",
            "Source is saved unmuted." if muted is False else "Source is muted or explicit unmuted state is missing.")
    volume = source.get("volume")
    valid_volume = type(volume) in (int, float) and math.isfinite(volume) and volume > 0
    finding("source_volume", "info" if valid_volume else "error",
            "Saved source volume is positive." if valid_volume else "Source volume is zero, negative, missing, or invalid.")
    finding("virtual_camera_audio", "info",
            "Virtual-camera video is distinct from BlackHole monitoring and Discord microphone selection; saved settings cannot establish remote audio or portrait acceptance.")
    return result


def audit_files(profile_path, scene_path, source_name, **kwargs):
    """Read only explicit profile/scene paths; return sanitized read/parse errors."""
    try:
        profile = Path(profile_path).read_text(encoding="utf-8-sig")
        scene = json.loads(Path(scene_path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, ValueError):
        return dict(recorded_config_only=True, runtime_verified=False,
                    remote_verified=False, read_only=True,
                    findings=[dict(code="input_unreadable", severity="error",
                                   message="Explicit profile or scene file could not be read/parsed.")])
    return audit(profile, scene, source_name, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--source-name", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(audit_files(args.profile, args.scene, args.source_name), indent=2))


if __name__ == "__main__":
    main()
