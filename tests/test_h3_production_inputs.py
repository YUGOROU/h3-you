"""All media are tiny fake bytes created here; no original recordings accessed."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from research.h3_personal import component_encode, freeze_dataset as freeze
from research.h3_personal import production_inputs as inputs


class ProductionInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('sources', 'media/clips', 'media/audio', 'references', 'code'):
            (self.root / name).mkdir(parents=True)
        repo = Path(__file__).resolve().parents[1]
        for name in inputs.CODE_PINS:
            source = repo / 'research' / name if name.startswith('hf_') else repo / 'research/h3_personal' / name
            shutil.copyfile(source, self.root / 'code' / name)
        self.source_map = {}
        for i in range(6):
            name = f'source{i}.MOV'
            path = self.root / 'sources' / name
            path.write_bytes(f'fake MOV {i}'.encode())
            self.source_map[freeze.file_sha256(path)] = name
        digests = list(self.source_map)
        selected = [dict(id=f's{i:03}', source=f'/never-access-original/source{i % 6}.MOV',
                         source_sha256=digests[i % 6], session_id=f'session{i % 6}',
                         start_seconds=i * 6, end_seconds=i * 6 + 124 / 24,
                         pool='identity' if i < 590 else 'motion',
                         human_review='approved_sampled_images') for i in range(715)]
        selection = freeze.build(selected, verify_sources=False)
        selection['source_bytes_verified'] = True
        self.write('selection', selection)
        raw = []
        for row in selection['samples']:
            record = copy.deepcopy(row)
            for kind, directory, suffix in (('video', 'clips', '.mp4'), ('audio', 'audio', '.wav')):
                name = f'{directory}/{row["id"]}{suffix}'
                path = self.root / 'media' / name
                path.write_bytes((kind + row['id']).encode())
                record[kind] = name
                record[kind + '_sha256'] = freeze.file_sha256(path)
            record['input_provenance'] = {k: row[k] for k in ('source_sha256', 'start_seconds', 'end_seconds')}
            raw.append(record)
        self.write('manifest', dict(schema='h3-production-raw-media-v1',
                                    dataset_sha256=selection['dataset_sha256'], samples=raw))
        self.write('summary', dict(dataset_sha256=selection['dataset_sha256'],
                                  input_revision='a' * 40,
                                  freeze_sha256=freeze.file_sha256(self.root / 'selection.json'),
                                  helper_sha256=inputs.HELPER_PINS, code_sha256=inputs.CODE_PINS,
                                  complete_raw_preprocessing=True, processed_clips=715))
        self.write('source_map', self.source_map)
        image = self.root / 'references' / 'approved.jpeg'
        image.write_bytes(b'fake explicitly approved image bytes')
        approval = dict(approved=True, approved_by='fixture operator', provenance={'source': 'fixture decisions.json'})
        self.write('reference_mapping', dict(schema='h3-production-reference-mapping-v1', **approval,
                   samples={r['id']: dict(path=image.name, sha256=freeze.file_sha256(image)) for r in raw}))
        self.write('caption_mapping', dict(schema='h3-production-caption-mapping-v1', **approval,
                   scope='session', captions={f'session{i}': f'Explicit fixture caption {i}' for i in range(6)}))
        self.args = {key + '_path': self.root / (key + '.json') for key in
                     ('selection', 'manifest', 'summary', 'source_map', 'reference_mapping', 'caption_mapping')}
        self.args.update(source_root=self.root / 'sources', media_root=self.root / 'media',
                         reference_root=self.root / 'references', code_root=self.root / 'code',
                         expected_input_revision='a' * 40)

    def write(self, name, value):
        (self.root / (name + '.json')).write_text(json.dumps(value))

    def mutate(self, name, change):
        value = json.loads((self.root / (name + '.json')).read_text())
        change(value)
        self.write(name, value)

    def reject(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            inputs.stage(**self.args)

    def test_rows_and_plan_and_hash_dedup(self):
        real_sha = freeze.file_sha256
        with patch.object(freeze, 'file_sha256', wraps=real_sha) as hashing:
            rows, receipt = inputs.stage(**self.args)
        calls = [str(call.args[0]) for call in hashing.call_args_list]
        self.assertEqual(len(calls), len(set(calls)))
        self.assertEqual(len(rows), 715)
        self.assertEqual(rows[-1]['quality_pool'], 'motion')
        self.assertEqual(rows[0]['caption'], 'Explicit fixture caption 0')
        self.assertEqual(receipt['reference_mapping']['provenance']['source'], 'fixture decisions.json')
        self.assertFalse(receipt['authenticated_provenance'])
        plan = component_encode.make_plan(rows, {k: 'b' * 64 for k in component_encode.cache.KINDS}, attested_by='fixture')
        self.assertEqual([r['id'] for r in plan['samples']], [r['id'] for r in rows])
        self.assertEqual(plan['samples'][-1]['quality_pool'], 'motion')

    def test_summary_pins_and_completion(self):
        original = json.loads((self.root / 'summary.json').read_text())
        for key, value in [('complete_raw_preprocessing', False), ('processed_clips', 714),
                           ('input_revision', 'b' * 40), ('freeze_sha256', 'c' * 64),
                           ('dataset_sha256', 'd' * 64), ('code_sha256', {}), ('helper_sha256', {})]:
            with self.subTest(key=key):
                self.write('summary', dict(original, **{key: value}))
                self.reject()
        self.write('summary', original)
        (self.root / 'code/preprocess.py').write_bytes(b'changed producer')
        self.reject()

    def test_order_and_source_range_proofs(self):
        original = json.loads((self.root / 'manifest.json').read_text())
        changes = [lambda d: d['samples'].reverse(),
                   lambda d: d['samples'].pop(),
                   lambda d: d.update(schema='legacy'),
                   lambda d: d['samples'][0].update(source_sha256='e' * 64),
                   lambda d: d['samples'][0].update(start_seconds=1),
                   lambda d: d['samples'][0].update(quality_pool='motion'),
                   lambda d: d['samples'][0]['input_provenance'].update(end_seconds=9)]
        for change in changes:
            value = copy.deepcopy(original)
            change(value)
            self.write('manifest', value)
            self.reject()

    def test_frozen_semantics(self):
        self.mutate('selection', lambda d: d.update(dataset_sha256='e' * 64))
        self.reject()

    def test_all_mounted_byte_classes(self):
        paths = [self.root / 'sources' / next(iter(self.source_map.values())),
                 self.root / 'media/clips/s000.mp4', self.root / 'media/audio/s000.wav',
                 self.root / 'references/approved.jpeg']
        for path in paths:
            with self.subTest(path=path):
                old = path.read_bytes()
                path.write_bytes(b'changed')
                self.reject()
                path.write_bytes(old)

    def test_reference_coverage_approval_and_paths(self):
        original = json.loads((self.root / 'reference_mapping.json').read_text())
        changes = [lambda d: d['samples'].pop('s000'),
                   lambda d: d['samples'].update(extra=d['samples']['s000']),
                   lambda d: d.update(approved=False), lambda d: d.update(provenance={})]
        for name in ('../references/approved.jpeg', '/tmp/approved.jpeg', './approved.jpeg', 'a\\b'):
            changes.append(lambda d, name=name: d['samples']['s000'].update(path=name))
        for change in changes:
            value = copy.deepcopy(original)
            change(value)
            self.write('reference_mapping', value)
            self.reject()

    def test_symlink_escapes(self):
        outside = self.root / 'outside.jpeg'
        outside.write_bytes(b'fake explicitly approved image bytes')
        (self.root / 'references/link.jpeg').symlink_to(outside)
        self.mutate('reference_mapping', lambda d: d['samples']['s000'].update(path='link.jpeg'))
        self.reject()

    def test_raw_path_escape(self):
        self.mutate('manifest', lambda d: d['samples'][0].update(video='../outside.mp4'))
        self.reject()

    def test_caption_exact_coverage_and_no_invention(self):
        original = json.loads((self.root / 'caption_mapping.json').read_text())
        for change in (lambda d: d['captions'].pop('session0'),
                       lambda d: d['captions'].update(session0=''),
                       lambda d: d.update(approved=False),
                       lambda d: d.update(scope='automatic')):
            value = copy.deepcopy(original)
            change(value)
            self.write('caption_mapping', value)
            self.reject()
        original.update(scope='sample', captions={f's{i:03}': f'Caption {i}' for i in range(715)})
        self.write('caption_mapping', original)
        rows, _ = inputs.stage(**self.args)
        self.assertEqual(rows[-1]['caption'], 'Caption 714')

    def test_cli_exclusive_stage(self):
        output = self.root / 'stage'
        argv = []
        for key, value in self.args.items():
            option = key.removesuffix('_path').replace('_', '-')
            argv.extend(['--' + option, str(value)])
        argv.extend(['--output', str(output)])
        self.assertEqual(inputs.main(argv), 0)
        rows = json.loads((output / 'rows.json').read_text())
        receipt = json.loads((output / 'receipt.json').read_text())
        self.assertEqual(len(rows), 715)
        self.assertEqual(receipt['rows_sha256'], inputs.hashlib.sha256(freeze.canonical(rows)).hexdigest())
        with self.assertRaises(FileExistsError):
            inputs.main(argv)
        self.mutate('summary', lambda d: d.update(complete_raw_preprocessing=False))
        argv[-1] = str(self.root / 'invalid-stage')
        with self.assertRaises(ValueError):
            inputs.main(argv)
        self.assertFalse((self.root / 'invalid-stage').exists())

    def test_source_mapping_exact_and_safe(self):
        original = dict(self.source_map)
        for mapping in ({}, dict(original, extra='extra.MOV'),
                        dict(original, **{next(iter(original)): '../outside.MOV'})):
            self.write('source_map', mapping)
            self.reject()

    def test_duplicate_json_keys(self):
        (self.root / 'source_map.json').write_text('{"same":1,"same":2}')
        self.reject()


if __name__ == '__main__':
    unittest.main()
