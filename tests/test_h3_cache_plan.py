"""Synthetic byte fixtures test planning, NOT real Trainer cache compatibility."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from research.h3_personal.cache_plan import (
    COMPONENTS, component_fingerprints, file_sha256, plan_cache_reuse,
)


class CachePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('video.mp4', 'audio.wav', 'portrait.png', 'other.png'):
            (self.root / name).write_bytes(name.encode())
        self.row = {'id': 'clip-01', 'video': 'video.mp4', 'audio': 'audio.wav',
                    'caption': 'Person speaking.', 'reference_image': 'portrait.png'}
        self.settings = {
            'geometry': {'width': 768, 'height': 1344, 'num_frames': 124, 'fps': 24},
            'encode_audio': True, 'references': True, 'reference_canvas': 'native',
            'lora_trigger': None, 'keyframes': [],
            'recipes': {component: {'model_revision': 'a' * 40,
                                    'trainer_revision': 'b' * 40,
                                    'diffusers_revision': 'c' * 40,
                                    'recipe': {'dtype': 'float32', 'fixture': component}}
                        for component in COMPONENTS},
        }
        self.cache = self.root / 'cache'
        self.cache.mkdir()

    def keys(self, row=None, settings=None):
        return component_fingerprints(row or self.row, settings or self.settings, self.root)

    def plan(self, rows=None, **kwargs):
        return plan_cache_reuse(rows if rows is not None else [self.row], self.settings,
                                base_dir=self.root, **kwargs)

    def changed(self, old, new):
        return {key for key in COMPONENTS if old[key] != new[key]}

    def sidecar(self, components=COMPONENTS):
        entries = []
        for component in components:
            artifact = self.cache / (component + '.safetensors')
            artifact.write_bytes(b'synthetic artifact ' + component.encode())
            entries.append({'component': component, 'fingerprint': self.keys()[component],
                            'path': artifact.name, 'sha256': file_sha256(artifact)})
        payload = {'schema_version': 1, 'artifacts': entries}
        (self.cache / 'provenance.json').write_text(json.dumps(payload))
        return payload

    def test_component_granular_media_changes(self):
        initial = self.keys()
        for name, expected in [('video.mp4', {'video'}), ('audio.wav', {'audio'}),
                               ('portrait.png', {'reference', 'text'})]:
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            self.assertEqual(self.changed(initial, self.keys()), expected)
            path.write_bytes(original)
        row = {**self.row, 'caption': 'New caption'}
        self.assertEqual(self.changed(initial, self.keys(row)), {'text'})

    def test_embedded_audio_depends_on_video(self):
        row = dict(self.row)
        del row['audio']
        before = self.keys(row)
        (self.root / 'video.mp4').write_bytes(b'new video')
        self.assertEqual(self.changed(before, self.keys(row)), {'video', 'audio'})

    def test_paths_ids_order_do_not_change_content_keys(self):
        (self.root / 'copy.mp4').write_bytes((self.root / 'video.mp4').read_bytes())
        row = {**self.row, 'id': 'renamed', 'video': str(self.root / 'copy.mp4')}
        self.assertEqual(self.keys(), self.keys(row))
        settings = dict(reversed(list(self.settings.items())))
        self.assertEqual(self.keys(), self.keys(settings=settings))

    def test_reference_order_is_significant(self):
        row = {**self.row, 'reference_image': ['portrait.png', 'other.png']}
        reversed_row = {**row, 'reference_image': list(reversed(row['reference_image']))}
        self.assertEqual(self.changed(self.keys(row), self.keys(reversed_row)), {'reference', 'text'})

    def test_recipe_trigger_geometry_invalidation(self):
        before = self.keys()
        for component in COMPONENTS:
            settings = copy.deepcopy(self.settings)
            settings['recipes'][component]['recipe']['dtype'] = 'float16'
            expected = {component, 'text'} if component == 'reference' else {component}
            self.assertEqual(self.changed(before, self.keys(settings=settings)), expected)
        settings = copy.deepcopy(self.settings)
        settings['geometry']['width'] = 1024
        self.assertEqual(self.changed(before, self.keys(settings=settings)), {'video'})
        settings = copy.deepcopy(self.settings)
        settings['geometry']['num_frames'] = 141
        self.assertEqual(self.changed(before, self.keys(settings=settings)), {'video', 'audio'})
        settings = copy.deepcopy(self.settings)
        settings['lora_trigger'] = 'person'
        self.assertEqual(self.changed(before, self.keys(settings=settings)), {'text'})
        settings = copy.deepcopy(self.settings)
        settings['recipes']['video']['model_revision'] = 'd' * 40
        self.assertEqual(self.changed(before, self.keys(settings=settings)), {'video'})

    def test_reference_audio_duration_reaches_text(self):
        row = {**self.row, 'reference_audio': 'audio.wav'}
        settings = copy.deepcopy(self.settings)
        settings['geometry']['num_frames'] = 141
        self.assertEqual(self.changed(self.keys(row), self.keys(row, settings)),
                         {'video', 'audio', 'reference', 'text'})

    def test_reference_video_geometry_reaches_text(self):
        row = {**self.row, 'reference_video': 'video.mp4'}
        settings = copy.deepcopy(self.settings)
        settings['geometry']['num_frames'] += 17
        self.assertEqual(self.changed(self.keys(row), self.keys(row, settings)), set(COMPONENTS))

    def test_disabled_components_not_scheduled(self):
        settings = copy.deepcopy(self.settings)
        settings.update(encode_audio=False, references=False)
        row = {**self.row, 'audio': 'missing.wav', 'reference_image': 'missing.png'}
        plan = plan_cache_reuse([row], settings, base_dir=self.root)
        for component in ('audio', 'reference'):
            self.assertEqual(plan['samples'][0]['components'][component]['action'], 'not_required')

    def test_missing_provenance_legacy_cache_fails_closed(self):
        (self.cache / 'index.json').write_text(json.dumps({
            'version': 1, 'samples': [{'id': self.row['id'], 'text_rows': 100}],
        }))
        (self.cache / 'video.safetensors').write_bytes(b'existing')
        plan = self.plan(cache_root=self.cache)
        self.assertTrue(all(item['action'] == 'compute'
                            for item in plan['samples'][0]['components'].values()))
        self.assertFalse(plan['tensor_validation_performed'])
        self.assertIn('provenance', plan['diagnostics'][0])

    def test_valid_sidecar_and_hash_reuses_across_ids(self):
        self.sidecar()
        rows = [self.row, {**self.row, 'id': 'other'}]
        plan = self.plan(rows, cache_root=self.cache)
        self.assertEqual(plan['shards'], [[], []])
        for sample in plan['samples']:
            self.assertTrue(all(item['action'] == 'reuse' for item in sample['components'].values()))

    def test_partial_cache_corruption_only_invalidates_component(self):
        self.sidecar()
        (self.cache / 'audio.safetensors').write_bytes(b'corrupt')
        plan = self.plan(cache_root=self.cache)
        actions = {key: value['action'] for key, value in plan['samples'][0]['components'].items()}
        self.assertEqual(actions, {'video': 'reuse', 'audio': 'compute',
                                   'reference': 'reuse', 'text': 'reuse'})
        (self.cache / 'video.safetensors').unlink()
        self.assertEqual(self.plan(cache_root=self.cache)['samples'][0]['components']['video']['action'], 'compute')

    def test_untrusted_artifact_paths_and_missing_hashes(self):
        payload = self.sidecar(['video'])
        outside = self.root / 'outside.safetensors'
        outside.write_bytes(b'outside')
        link = self.cache / 'link.safetensors'
        link.symlink_to(outside)
        for path in ('../outside.safetensors', str(outside), 'link.safetensors'):
            with self.subTest(path=path):
                entry = {**payload['artifacts'][0], 'path': path, 'sha256': file_sha256(outside)}
                (self.cache / 'provenance.json').write_text(json.dumps({
                    'schema_version': 1, 'artifacts': [entry]}))
                self.assertEqual(self.plan(cache_root=self.cache)['samples'][0]['components']['video']['action'], 'compute')
        del payload['artifacts'][0]['sha256']
        (self.cache / 'provenance.json').write_text(json.dumps(payload))
        self.assertEqual(self.plan(cache_root=self.cache)['samples'][0]['components']['video']['action'], 'compute')

    def test_bad_sidecar_is_not_evidence(self):
        for content in ('broken', 'null', '[]', '{"schema_version": 2}',
                        '{"schema_version": 1, "artifacts": [null, {}]}'):
            (self.cache / 'provenance.json').write_text(content)
            self.assertTrue(self.plan(cache_root=self.cache)['diagnostics'])

    def test_deterministic_disjoint_max_two_shards_and_deduplication(self):
        rows = [{**self.row, 'id': f'clip-{index}', 'caption': f'Caption {index % 2}'}
                for index in range(9)]
        first = self.plan(rows)
        self.assertEqual(first, self.plan(list(reversed(rows))))
        left, right = first['shards']
        keys = lambda shard: {(item['component'], item['fingerprint']) for item in shard}
        self.assertFalse(keys(left) & keys(right))
        self.assertEqual(len(left) + len(right), 5)  # 1 video/audio/reference + 2 texts
        self.assertLessEqual(abs(len(left) - len(right)), 1)
        self.assertEqual(sum(len(item['sample_ids']) for item in left + right), 9 * 4)
        self.assertEqual(len(self.plan(rows, shards=1)['shards']), 1)
        for value in (0, 3, True, 1.5):
            with self.assertRaises(ValueError):
                self.plan(rows, shards=value)
        self.assertEqual(self.plan([])['shards'], [[], []])

    def test_missing_settings_inputs_and_unsupported_options_fail_closed(self):
        for key in self.settings:
            settings = copy.deepcopy(self.settings)
            del settings[key]
            with self.assertRaises(ValueError):
                self.keys(settings=settings)
        for revision in ('main', '', None):
            settings = copy.deepcopy(self.settings)
            settings['recipes']['text']['model_revision'] = revision
            with self.assertRaises(ValueError):
                self.keys(settings=settings)
        settings = copy.deepcopy(self.settings)
        settings['keyframes'] = ['first_frame']
        with self.assertRaises(ValueError):
            self.keys(settings=settings)
        with self.assertRaises(FileNotFoundError):
            self.keys({**self.row, 'video': 'missing'})
        with self.assertRaises(ValueError):
            self.keys({**self.row, 'reference_images': ['other.png']})
        for rows in ([self.row, self.row], [{**self.row, 'id': '../escape'}],
                     [{key: value for key, value in self.row.items() if key != 'id'}]):
            with self.assertRaises(ValueError):
                self.plan(rows)

    def test_cli_local_stdout_and_no_cache_writes(self):
        manifest = self.root / 'manifest.json'
        settings = self.root / 'settings.json'
        manifest.write_text(json.dumps([self.row]))
        settings.write_text(json.dumps(self.settings))
        before = {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        result = subprocess.run([sys.executable, '-m', 'research.h3_personal.cache_plan',
                                 '--manifest', str(manifest), '--settings', str(settings),
                                 '--cache-root', str(self.cache)], check=True,
                                capture_output=True, text=True)
        self.assertTrue(json.loads(result.stdout)['local_only'])
        self.assertEqual(before, {path: path.read_bytes() for path in self.root.rglob('*') if path.is_file()})


if __name__ == '__main__':
    unittest.main()
