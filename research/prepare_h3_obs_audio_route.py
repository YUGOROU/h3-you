"""Offline candidate only: never apply an OBS profile or access devices.

Later application requires separate permission; live OBS may overwrite edits.
"""
import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from research.h3_personal.obs_delivery_audit import audit

KEYS = {"monitoringdeviceid": ("MonitoringDeviceId", "BlackHole2ch_UID"),
        "monitoringdevicename": ("MonitoringDeviceName", "BlackHole 2ch")}


def repair_profile(text):
    """Preserve every noncritical character; reject uncertain saved routes."""
    if not isinstance(text, str) or '\x00' in text:
        raise ValueError('Invalid profile')
    lines = text.splitlines(keepends=True)
    newline = '\r\n' if '\r\n' in text else '\n'
    section = None
    audio_end = None
    removed = set()
    previous_indent = None
    previous_critical = False
    audio_count = 0
    for index, line in enumerate(lines):
        stripped = line.strip().lstrip('\ufeff')
        if not stripped or stripped.startswith(('#', ';')):
            continue
        indent = len(line) - len(line.lstrip())
        match = re.match(r'([^:=]+?)\s*[:=](.*)$', stripped)
        key = match.group(1).strip().lower() if match else None
        critical = key in KEYS
        if previous_indent is not None and indent > previous_indent:
            if previous_critical or critical or stripped.startswith('['):
                raise ValueError('Ambiguous critical continuation')
            continue
        if stripped.startswith('['):
            header = re.fullmatch(r'\[([^\[\]]+)\]', stripped)
            if not header:
                raise ValueError('Malformed section header')
            if section == 'Audio':
                audio_end = index
            section = header.group(1)
            if section.strip().lower() == 'audio':
                if section != 'Audio' or audio_count:
                    raise ValueError('Ambiguous Audio section')
                audio_count += 1
            previous_indent = None
            previous_critical = False
            continue
        if critical:
            if section != 'Audio':
                raise ValueError('Critical key outside Audio')
            value = match.group(2).strip()
            allowed = {'', 'default', 'devicedefault', KEYS[key][1].lower()}
            if key=='monitoringdevicename':
                allowed.add('既定')  # OBS Japanese localized Default display name.
            if value.lower() not in allowed:
                raise ValueError('Unsupported existing monitoring destination')
            removed.add(index)
        previous_indent = indent
        previous_critical = critical
    if section == 'Audio':
        audio_end = len(lines)
    if not audio_count:
        raise ValueError('Explicit Audio section required')
    canonical = ''.join(f'{name}={value}{newline}' for name, value in KEYS.values())
    before = ''.join(line for i, line in enumerate(lines[:audio_end]) if i not in removed)
    if before and not before.endswith(('\n', '\r')):
        before += newline
    candidate = before + canonical + ''.join(lines[audio_end:])
    # Preserve unrelated OBS-written duplicates byte-for-byte, especially the
    # accepted geometry. This parser checks syntax only, not their precedence.
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    try:
        parser.read_string(candidate.lstrip('\ufeff'))
    except configparser.Error:
        raise ValueError('Invalid canonical profile') from None
    if (parser.get('Audio','MonitoringDeviceId',raw=True)!='BlackHole2ch_UID' or
            parser.get('Audio','MonitoringDeviceName',raw=True)!='BlackHole 2ch'):
        raise ValueError('Canonical audio route not verified')
    return candidate


def _read_exact(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink inputs are not allowed')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Regular input files required')
        return stream.read()


def prepare(sourceprofile, scene, source_name, directory):
    """Read two explicit files and create a fresh private external review folder."""
    profile_path, scene_path = Path(sourceprofile).absolute(), Path(scene).absolute()
    target = Path(directory).absolute()
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError('Symlink output is not allowed')
    for path in (profile_path, scene_path):
        if target == path.parent or path.parent in target.parents:
            raise ValueError('Review directory must be external to input directories')
    original = _read_exact(profile_path)
    scene_bytes = _read_exact(scene_path)
    try:
        candidate = repair_profile(original.decode('utf-8')).encode('utf-8')
        scene_data = json.loads(scene_bytes.decode('utf-8-sig'))
    except (UnicodeError, ValueError):
        raise ValueError('Invalid profile or scene input') from None
    result = audit(candidate.decode('utf-8').lstrip('\ufeff'), scene_data, source_name)
    # Fixed auditor codes/severities only: no arbitrary paths, settings or names.
    sanitized = {item['code']: item['severity'] for item in result['findings']}
    plan = dict(offline_only=True, applied=False, runtime_verified=False,
                separate_apply_permission_required=True, live_obs_may_overwrite=True,
                critical_audio_keys_canonical=True, unrelated_settings_preserved_unverified=True,
                source_sha256=hashlib.sha256(original).hexdigest(),
                candidate_sha256=hashlib.sha256(candidate).hexdigest(),
                audit=sanitized,
                review_changes={'MonitoringDeviceId': 'BlackHole2ch_UID',
                                'MonitoringDeviceName': 'BlackHole 2ch'})
    target.mkdir(mode=0o700, exist_ok=False)
    for name, content in [('basic.ini.candidate', candidate),
                          ('plan.json', (json.dumps(plan, indent=2)+'\n').encode())]:
        fd = os.open(target/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('profile', 'scene', 'source', 'directory'):
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args(argv)
    try:
        prepare(args.profile, args.scene, args.source, args.directory)
    except (OSError, ValueError):
        parser.exit(1, 'Offline preparation rejected; no profile applied.\n')
    print('Offline review candidate prepared; no profile applied. Separate apply permission required.')


if __name__ == '__main__':
    main()
