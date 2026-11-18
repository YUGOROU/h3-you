"""Offline launch/staging contracts; no Hub/device/model/runtime operations."""
from copy import deepcopy
from dataclasses import asdict
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock,patch

from research import launch_h3_direct_conversation as launch
from research import launch_h3_live_voice as shared
from research.h3_personal.native_tts_backend import config_sha256
from test_h3_native_tts_boot_wiring import native_spec


def fake_prepare(stage,*args,**kwargs):
    # Executed through the real launcher's preparation FunctionType adapter.
    stage.mkdir()
    value=native_spec()
    (stage/'spec.json').write_text(json.dumps(value))


class NativeLaunchWiring(unittest.TestCase):
    def test_ready_binds_native_selection_and_expiry(self):
        config=native_spec()['live_direct_session']['native_tts']
        config['expires_at']=time.time()+600
        row=dict(type='h3_direct_ready',status='ready',session='a'*32,profile_sha256='b'*64,
                 epoch=1,max_segments=16,max_total_frames=1654,port=7860,
                 expires_at=config['expires_at'],tts_mode='native_voxcpm',
                 native_tts_config_sha256=config_sha256(config))
        self.assertIs(launch.validate_ready(row,'a'*32,tts_mode='native_voxcpm',native_tts=config),row)
        for changed in ({'native_tts_config_sha256':'0'*64},{'tts_mode':'complete'},
                        {'expires_at':config['expires_at']+1}):
            with self.subTest(changed=changed),self.assertRaises(ValueError):
                launch.validate_ready(dict(row,**changed),'a'*32,tts_mode='native_voxcpm',native_tts=config)
        with self.assertRaises(ValueError):launch.validate_ready(row,'a'*32)

    def test_rtc_ready_is_explicit_and_boot_bound(self):
        config=native_spec()['live_direct_session']['native_tts'];config['expires_at']=time.time()+600
        row=dict(type='h3_direct_ready',status='ready',session='a'*32,profile_sha256='b'*64,
                 epoch=1,max_segments=16,max_total_frames=1654,port=7860,
                 expires_at=config['expires_at'],tts_mode='native_voxcpm',
                 native_tts_config_sha256=config_sha256(config),media_transport='webrtc-rtp-v1',
                 rtc_boot='c'*32,ice_profile='hf-cloudflare-turn-tls-v1')
        opts=dict(tts_mode='native_voxcpm',native_tts=config,media_transport='webrtc-rtp-v1')
        self.assertIs(launch.validate_ready(row,'a'*32,**opts),row)
        for change in ({'rtc_boot':12345678901234567890123456789012},
                       {'ice_profile':'host-only-v1'},{'media_transport':'http-paired-v1'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                launch.validate_ready(dict(row,**change),'a'*32,**opts)
        with self.assertRaises(ValueError):launch.validate_ready(row,'a'*32,tts_mode='native_voxcpm',native_tts=config)

    def test_rtc_stage_includes_only_isolated_runtime_lock_files(self):
        row=native_spec()['live_direct_session'];row.update(media_transport='webrtc-rtp-v1',ice_profile='hf-cloudflare-turn-tls-v1')
        with tempfile.TemporaryDirectory() as root, \
                patch('research.prepare_h3_fast_ref2va_gpu.prepare',fake_prepare), \
                patch.object(launch,'first_party_sources',return_value=[]), \
                patch.object(launch,'closure'),patch.object(launch,'rewrite_allowlist'),patch.object(launch,'inspect'):
            stage=launch.prepare(Path(root)/'owner',row,Mock())
            for name in ('uv.lock','pyproject.toml'):
                self.assertEqual((stage/'research/webrtc_experiment'/name).read_bytes(),
                                 (launch.ROOT/'research/webrtc_experiment'/name).read_bytes())
            self.assertFalse((stage/'research/webrtc_experiment/.venv').exists())

    def test_shared_native_inspect_never_reads_irodori_manifest(self):
        value=native_spec();value['config']={'timeout_seconds':1200}
        inventory={'files':[{'path':path} for path in shared.EXTRA]}
        with patch.object(shared,'base_inspect',return_value=(value,inventory)), \
                patch.object(Path,'read_text',side_effect=AssertionError('local Irodori file forbidden')):
            self.assertEqual(shared.inspect(Path('/unused')),(value,inventory))
        invalid=deepcopy(inventory);invalid['files'].append({'path':'tts-manifest.json'})
        with patch.object(shared,'base_inspect',return_value=(value,invalid)):
            with self.assertRaises(ValueError):shared.inspect(Path('/unused'))

    def test_native_pre_submit_skips_irodori_models(self):
        value=native_spec();value['live_direct_session']['native_tts']['expires_at']=time.time()+600
        api=Mock()
        with tempfile.TemporaryDirectory() as root,patch.object(shared,'base_pre_submit') as base:
            shared.pre_submit(api,value,Path(root))
            base.assert_called_once_with(api,value,Path(root))
            self.assertEqual(api.mock_calls,[])
            (Path(root)/'tts-manifest.json').write_text('{}')
            with self.assertRaises(ValueError):shared.pre_submit(api,value,Path(root))

    def test_expired_native_lease_refused_before_provider_checks(self):
        value=native_spec()
        with patch.object(shared,'base_pre_submit') as base:
            with self.assertRaises(ValueError):shared.pre_submit(Mock(),value,Path('/unused'))
            base.assert_not_called()

    def test_native_prepare_omits_irodori_manifest(self):
        row=native_spec()['live_direct_session']
        with tempfile.TemporaryDirectory() as root, \
                patch('research.prepare_h3_fast_ref2va_gpu.prepare',fake_prepare), \
                patch.object(launch,'PRIVATE',Path(root)/'absent-irodori-inputs'), \
                patch.object(launch,'first_party_sources',return_value=[]), \
                patch.object(launch,'closure'),patch.object(launch,'rewrite_allowlist'), \
                patch.object(launch,'inspect'):
            stage=launch.prepare(Path(root)/'stage-owner',row,Mock())
            self.assertFalse((stage/'tts-manifest.json').exists())
            self.assertEqual(json.loads((stage/'spec.json').read_text())['live_direct_session'],row)

    def test_native_cli_offline_does_not_launch_or_require_secret(self):
        value=native_spec()['live_direct_session']['native_tts']
        with tempfile.TemporaryDirectory() as root,patch.object(launch,'run',side_effect=AssertionError('GPU forbidden')):
            path=Path(root)/'config.json';path.write_text(json.dumps(value))
            with contextlib.redirect_stdout(io.StringIO()):
                launch.main(['--directory',str(Path(root)/'session'),'--native-tts-config',str(path)])
            self.assertFalse((Path(root)/'session').exists())


if __name__=='__main__':unittest.main()
