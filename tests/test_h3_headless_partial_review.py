"""Synthetic CPU/FFmpeg tests only; no private media/provider/device access."""
from dataclasses import replace
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from research.h3_personal.headless_av_sink import HeadlessAVSink, ReviewArchive
from research.h3_personal.paired_av import (
    AudioRegion, LiveAVBinding, PresentationCut, ReceivedAVLease, encode,
)
from research.h3_personal.paired_av_wire import envelope_record


def sha(body):
    return hashlib.sha256(body).hexdigest()


def ff(*args, data=None):
    return subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', *args],
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True, timeout=15).stdout


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg required')
class PartialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.binding = LiveAVBinding('synthetic-review', 'a'*64, 17,
                                    max_segments=8, max_total_frames=100)
        self.video = self.root/'full.mp4'
        self.make_video(self.video, 6)
        self.video_body = self.video.read_bytes()
        self.old = struct.pack('<h', -1234)*4000
        self.new = b''.join(struct.pack('<h', (i*11)%20000-10000) for i in range(8000))
        self.pcm = self.old+self.new
        regions = (
            AudioRegion('speech', 20000, 24000, 'old', 1, 100000, sha(self.old)),
            AudioRegion('speech', 24000, 32000, 'new', 2, 0, sha(self.new)),
        )
        self.envelope = encode(self.binding, 1, 10, 16, self.video_body,
            self.pcm, regions, False, presentation_epoch=1,
            presentation_cut=PresentationCut(12, 0, 1))
        self.lease = ReceivedAVLease(self.envelope, self.video_body, self.pcm)

    def tearDown(self):
        self.temp.cleanup()

    def make_video(self, path, count):
        frames = b''.join(bytes((i*31, 255-i*23, i*13))*16*16 for i in range(count))
        ff('-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', '16x16',
           '-framerate', '24', '-i', 'pipe:0', '-frames:v', str(count),
           '-c:v', 'libx264', '-threads', '1', '-crf', '0', '-pix_fmt', 'yuv420p',
           str(path), data=frames)

    def sink(self, **kwargs):
        return HeadlessAVSink(self.root, self.binding, width=16, height=16,
            discontinuity_origin=(1, 12), max_total_frames=100, **kwargs)

    def test_full_decode_then_exact_paired_suffix_preserves_new_source_head(self):
        full_rgb = ff('-i', str(self.video), '-map', '0:v', '-pix_fmt', 'rgb24',
                      '-f', 'rawvideo', 'pipe:1')
        with self.sink() as sink:
            result = sink.accept(self.video.name, self.lease, presentation_start_frame=12)
            self.assertEqual((result.frame_start, result.frame_stop), (12, 16))
            self.assertEqual((result.sample_start, result.sample_stop), (24000, 32000))
            self.assertEqual(result.omitted_frame_range, (10, 12))
            self.assertEqual(result.disposition, 'partially-presented-interrupted')
            self.assertEqual(result.regions[0].source_offset, 0)
            self.assertEqual(result.regions[0].source_sha256, sha(self.new))
            output = sink.finish()
            decoded = ff('-i', str(output), '-map', '0:a', '-f', 's16le', 'pipe:1')
            rgb = ff('-i', str(output), '-map', '0:v', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1')
            self.assertEqual(decoded, self.new)
            self.assertEqual(rgb, full_rgb[2*16*16*3:])
            self.assertEqual(sink.audio_evidence['source_pcm_samples'], 8000)
            event = sink.timing['events'][0]
            self.assertEqual(event['verified_full_frame_range'], (10, 16))
            self.assertEqual(event['omitted_frame_range'], (10, 12))
            self.assertAlmostEqual(event['media_seconds'], 4/24)
            # Original immutable lease remains the complete verified packet.
            self.assertEqual(self.lease.segment.frame_start, 10)
            self.assertEqual(self.lease.audio_pcm, self.pcm)
        self.assertFalse(output.exists())

    def test_suffix_then_next_full_packet_preserves_global_sequence_and_all_pcm(self):
        following = b''.join(struct.pack('<h', (i*19)%24000-12000) for i in range(12000))
        e = encode(self.binding, 2, 16, 22, self.video_body, following,
            (AudioRegion('speech', 32000, 44000, 'new', 2, 8000, sha(following)),),
            True, presentation_epoch=1)
        with self.sink() as sink:
            sink.accept(self.video.name, self.lease, presentation_start_frame=12)
            result = sink.accept(self.video.name, ReceivedAVLease(e, self.video_body, following))
            self.assertEqual((result.frame_start, result.frame_stop), (16, 22))
            self.assertEqual(sink._index, 3)
            out = sink.finish()
            self.assertEqual(ff('-i', str(out), '-map', '0:a', '-f', 's16le', 'pipe:1'),
                             self.new+following)
            self.assertEqual(sink.audio_evidence['source_pcm_samples'], 20000)
            self.assertEqual(sink.timing['global_frame_origin'], 12)

    def test_full_snapshot_hash_is_checked_before_suffix_decode(self):
        self.video.write_bytes(b'not-the-verified-video'+self.video_body)
        with self.sink() as sink:
            with self.assertRaises(ValueError):
                sink.accept(self.video.name, self.lease, presentation_start_frame=12)
            self.assertEqual(sink.audio_evidence['ack_count'], 0)

    def test_silence_partition_is_clipped_and_fresh_source_region_stays_exact(self):
        sound = self.new[2000*2:]
        pcm = bytes(6000*2)+sound
        regions = (AudioRegion('silence', 20000, 26000),
                   AudioRegion('speech', 26000, 32000, 'new', 2, 0, sha(sound)))
        e = encode(self.binding, 1, 10, 16, self.video_body, pcm, regions,
            False, presentation_epoch=1, presentation_cut=PresentationCut(12, 0, 1))
        with self.sink() as sink:
            result = sink.accept(self.video.name, ReceivedAVLease(e, self.video_body, pcm),
                                 presentation_start_frame=12)
            self.assertEqual(result.regions[0], AudioRegion('silence', 24000, 26000))
            self.assertEqual(result.regions[1].source_offset, 0)
            self.assertEqual(result.regions[1].source_sha256, sha(sound))
            out = sink.finish()
            self.assertEqual(ff('-i', str(out), '-map', '0:a', '-f', 's16le', 'pipe:1'),
                             pcm[4000*2:])

    def test_omitted_pcm_hash_is_still_verified_before_any_crop(self):
        corrupted = replace(self.lease, audio_pcm=b'\xff\xff'+self.pcm[2:])
        with self.sink() as sink:
            with self.assertRaisesRegex(ValueError, 'hash'):
                sink.accept(self.video.name, corrupted, presentation_start_frame=12)
            self.assertEqual(sink.audio_evidence['ack_count'], 0)

    def test_full_video_count_is_checked_not_just_requested_suffix_count(self):
        path = self.root/'extra.mp4'
        self.make_video(path, 7)
        video = path.read_bytes()
        e = replace(self.envelope, video_sha256=sha(video), video_size=len(video))
        with self.sink() as sink:
            with self.assertRaisesRegex(ValueError, 'frame count'):
                sink.accept(path.name, ReceivedAVLease(e, video, self.pcm),
                            presentation_start_frame=12)

    def test_cut_is_explicit_exact_and_requires_matching_global_origin(self):
        cases = ((None, self.envelope), (13, self.envelope),
                 (True, self.envelope),
                 (12, replace(self.envelope, presentation_cut=None)))
        for frame, envelope in cases:
            with self.subTest(frame=frame, cut=envelope.presentation_cut):
                with self.sink() as sink:
                    with self.assertRaises(ValueError):
                        sink.accept(self.video.name,
                            ReceivedAVLease(envelope, self.video_body, self.pcm),
                            presentation_start_frame=frame)
        with HeadlessAVSink(self.root, self.binding, width=16, height=16,
                            discontinuity_origin=(1, 10), max_total_frames=100) as sink:
            with self.assertRaisesRegex(ValueError, 'noncontiguous'):
                sink.accept(self.video.name, self.lease, presentation_start_frame=12)

    def test_finished_span_captured_once_before_default_temp_cleanup(self):
        archive = ReviewArchive(self.root/'review', deadline=time.monotonic()+20)
        with self.sink(review_archive=archive) as sink:
            sink.accept(self.video.name, self.lease, presentation_start_frame=12)
            output = sink.finish()
            original = output.read_bytes()
            self.assertTrue(output.exists())
            self.assertEqual(len(archive.snapshot()['entries']), 1)
            first = archive.snapshot()['entries'][0]['files'][0]
            self.assertEqual(first['method'], 'hard-link')
            self.assertIsNone(first['sha256'])
            self.assertEqual((archive.root/first['name']).stat().st_ino, output.stat().st_ino)
            self.assertEqual(sink.timing['review_retention']['method'], 'hard-link')
            self.assertIsNotNone(sink.timing['review_retention']['duration_seconds'])
        sink.close()
        self.assertFalse(output.exists())
        snap = archive.snapshot()
        self.assertEqual(len(snap['entries']), 1)
        row = snap['entries'][0]
        self.assertEqual(row['kind'], 'encoder-span')
        self.assertFalse(row['metadata']['interrupted'])
        self.assertEqual(row['metadata']['origin_frame'], 12)
        child = row['metadata']['child']
        self.assertFalse(child['device_output_verified'])
        self.assertFalse(child['all_buffered_audio_observed'])
        self.assertEqual(child['timing']['global_frame_origin'], 12)
        saved = archive.root/row['files'][0]['name']
        self.assertEqual(saved.read_bytes(), original)
        self.assertEqual(ff('-i', str(saved), '-map', '0:a', '-f', 's16le', 'pipe:1'), self.new)
        self.assertEqual(archive.root.stat().st_mode & 0o777, 0o700)
        self.assertTrue(all(p.stat().st_mode & 0o777 == 0o600 for p in archive.root.iterdir()))
        self.assertEqual(sum(p.stat().st_size for p in archive.root.iterdir()), snap['bytes_used'])

    def test_cancel_retains_only_existing_encoder_bytes_and_never_claims_device(self):
        archive = ReviewArchive(self.root/'cancel-review', deadline=time.monotonic()+20)
        sink = self.sink(review_archive=archive)
        sink.accept(self.video.name, self.lease, presentation_start_frame=12)
        end = time.monotonic()+2
        while sink.output.stat().st_size == 0 and time.monotonic() < end:
            time.sleep(.01)
        self.assertGreater(sink.output.stat().st_size, 0)
        output = sink.output
        sink.cancel(); sink.close()
        self.assertFalse(output.exists())
        rows = archive.snapshot()['entries']
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['metadata']['interrupted'])
        self.assertFalse(rows[0]['all_buffered_audio_observed'])
        self.assertTrue(all(p.poll() is not None for p in sink._processes))

    def test_span_cannot_be_archived_while_encoder_is_still_writing(self):
        archive = ReviewArchive(self.root/'not-live', deadline=time.monotonic()+20)
        with self.sink(review_archive=archive) as sink:
            sink.accept(self.video.name, self.lease, presentation_start_frame=12)
            self.assertIsNone(sink._encoder.poll())
            with self.assertRaisesRegex(RuntimeError, 'reaped encoder'):
                sink._capture_review_span(interrupted=True)
            self.assertFalse(sink._review_capture_attempted)
            self.assertEqual(archive.snapshot()['entries'], [])
            sink.finish()
        self.assertEqual(len(archive.snapshot()['entries']), 1)

    def test_pair_archive_preserves_full_reference_not_suffix_or_presentation_claim(self):
        archive = ReviewArchive(self.root/'pairs', deadline=time.monotonic()+20)
        archive.capture_pair(self.envelope, self.video_body, self.pcm)
        snap = archive.snapshot()
        row = snap['entries'][0]
        self.assertEqual(row['kind'], 'full-native-cut-reference')
        self.assertEqual((archive.root/'pair-001.mp4').read_bytes(), self.video_body)
        self.assertEqual((archive.root/'pair-001.s16le').read_bytes(), self.pcm)
        sidecar = json.loads((archive.root/'entry-000.json').read_text())
        self.assertEqual(sidecar['metadata']['envelope'], envelope_record(self.envelope))
        self.assertEqual(sidecar['metadata']['omitted_frame_range'], [10, 12])
        self.assertEqual(sidecar['metadata']['suffix_frame_range'], [12, 16])
        self.assertEqual(json.loads((archive.root/'review-index.json').read_text()), snap)
        self.assertEqual(sum(p.stat().st_size for p in archive.root.iterdir()), snap['bytes_used'])
        # Defensive snapshot and duplicate refusal cannot overwrite a capture.
        snap['entries'].clear()
        self.assertEqual(len(archive.snapshot()['entries']), 1)
        with self.assertRaises(FileExistsError):
            archive.capture_pair(self.envelope, self.video_body, self.pcm)
        self.assertEqual(len(archive.snapshot()['entries']), 1)
        with self.assertRaises(ValueError):
            archive.capture_pair(replace(self.envelope, presentation_cut=None), self.video_body, self.pcm)


class ArchiveBoundsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.source = self.root/'synthetic.nut'
        self.source.write_bytes(b'synthetic archive byte-bound fixture')

    def tearDown(self):
        self.temp.cleanup()

    def test_shared_budget_rolls_back_failed_capture_not_truncate(self):
        archive = ReviewArchive(self.root/'small', deadline=time.monotonic()+20, max_bytes=4096)
        archive.capture_span(self.source, (0, 0), True, {})
        before = archive.snapshot()
        # Captured/reaped source is immutable by contract; use a new owned inode.
        another = self.root/'another.nut'
        another.write_bytes(b'x'*4096)
        with self.assertRaisesRegex(ValueError, 'cap'):
            archive.capture_span(another, (1, 119), True, {})
        self.assertEqual(archive.snapshot(), before)
        self.assertEqual(sum(p.stat().st_size for p in archive.root.iterdir()), before['bytes_used'])
        self.assertFalse(any(p.name.startswith('.') for p in archive.root.iterdir()))

    def test_hard_link_is_fast_path_and_original_path_is_not_moved(self):
        archive = ReviewArchive(self.root/'link-copy', deadline=time.monotonic()+20)
        original = self.source.read_bytes()
        result = archive.capture_span(self.source, (0, 0), True, {})
        saved = result['files'][0]
        target = archive.root/saved['name']
        self.assertEqual(saved['method'], 'hard-link')
        self.assertIsNone(saved['sha256'])
        self.assertEqual(target.stat().st_ino, self.source.stat().st_ino)
        self.assertEqual(target.read_bytes(), original)
        self.assertTrue(self.source.exists())
        self.assertGreaterEqual(saved['duration_seconds'], 0)
        self.assertFalse(result['retention_time_is_gpu_generation'])
        self.source.unlink()
        self.assertEqual(target.read_bytes(), original)

    def test_cross_filesystem_copy_fallback_is_explicit_and_hashed(self):
        archive = ReviewArchive(self.root/'exdev', deadline=time.monotonic()+20)
        actual_link = os.link
        def different_device(source, target, **kwargs):
            if Path(source) == self.source:
                raise OSError(errno.EXDEV, 'synthetic cross-device link')
            return actual_link(source, target, **kwargs)
        with patch('research.h3_personal.headless_av_sink.os.link', side_effect=different_device):
            result = archive.capture_span(self.source, (0, 0), True, {})
        saved = result['files'][0]
        self.assertEqual(saved['method'], 'copy-exdev')
        self.assertEqual(saved['sha256'], sha(self.source.read_bytes()))
        self.assertGreaterEqual(saved['duration_seconds'], 0)
        self.assertNotEqual((archive.root/saved['name']).stat().st_ino, self.source.stat().st_ino)
        self.assertEqual(sum(p.stat().st_size for p in archive.root.iterdir()),
                         archive.snapshot()['bytes_used'])

    def test_finite_deadline_and_caps_are_enforced(self):
        for opts in (dict(deadline=float('inf')), dict(deadline=True),
                     dict(deadline=time.monotonic()-1),
                     dict(deadline=time.monotonic()+2000),
                     dict(deadline=time.monotonic()+20, max_bytes=0),
                     dict(deadline=time.monotonic()+20, max_bytes=True),
                     dict(deadline=time.monotonic()+20, max_bytes=2*1024**3+1)):
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                ReviewArchive(self.root/'invalid', **opts)
        archive = ReviewArchive(self.root/'expires', deadline=time.monotonic()+.05)
        time.sleep(.06)
        with self.assertRaises(TimeoutError):
            archive.capture_span(self.source, (0, 0), True, {})
        self.assertEqual(archive.snapshot()['entries'], [])

    def test_private_fresh_outside_repo_nonsymlink_directory(self):
        existing = self.root/'exists'; existing.mkdir()
        with self.assertRaises(FileExistsError):
            ReviewArchive(existing, deadline=time.monotonic()+20)
        link = self.root/'link'; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'nonsymlink'):
            ReviewArchive(link/'child', deadline=time.monotonic()+20)
        repo = Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(ValueError, 'outside'):
            ReviewArchive(repo/'forbidden-review-test-no-create', deadline=time.monotonic()+20)
        archive = ReviewArchive(self.root/'safe', deadline=time.monotonic()+20)
        alias = self.root/'source-link'; alias.symlink_to(self.source)
        with self.assertRaises(OSError):
            archive.capture_span(alias, (0, 0), True, {})
        self.assertEqual(archive.snapshot()['entries'], [])

    def test_metadata_is_json_bounded_without_repr_fallback(self):
        archive = ReviewArchive(self.root/'json', deadline=time.monotonic()+20)
        for bad in ({'bad': object()}, {'bad': float('nan')}, {'big': 'a'*(256*1024)}):
            with self.subTest(bad_type=type(next(iter(bad.values()))).__name__):
                with self.assertRaises((TypeError, ValueError)):
                    archive.capture_span(self.source, (0, 0), True, bad)
        self.assertEqual(archive.snapshot()['entries'], [])


if __name__ == '__main__':
    unittest.main()
