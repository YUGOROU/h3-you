"""CPU-only spatial/order regression through the production two-encode path.

Synthetic grayscale RGB only; no OBS instance, network, devices or model imports.
This checks codec transport, not actual OBS/Discord rendering acceptance.
"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from research.h3_personal.obs_av_sink import output_arguments
from research.h3_personal.realtime_segments import segment_timing
from research.run_h3_realtime_first import encode_rgb_segment


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),
                     'FFmpeg and ffprobe required')
class OBSDoubleEncodeSpatialTests(unittest.TestCase):
    WIDTH = 252
    HEIGHT = 448
    TILE_WIDTH = WIDTH // 4
    TILE_HEIGHT = HEIGHT // 4
    MAX_ERROR = 5

    @staticmethod
    def level(frame, row, column):
        # Distinct coordinate tiles and temporal states expose transpose, crop,
        # row-stride corruption, duplication and reordering (including reversal).
        return 24 + (row * 4 + column) * 8 + ((frame * 5) % 6) * 16

    def raw_frames(self, count):
        frames = []
        for frame in range(count):
            rows = []
            for row in range(4):
                scanline = b''.join(
                    bytes([self.level(frame, row, column)]) * (self.TILE_WIDTH * 3)
                    for column in range(4))
                rows.append(scanline * self.TILE_HEIGHT)
            frames.append(b''.join(rows))
        return b''.join(frames)

    def decode(self, path):
        return subprocess.run(
            ['ffmpeg', '-v', 'error', '-nostdin', '-i', str(path),
             '-map', '0:v:0', '-an', '-vsync', '0', '-threads', '1',
             '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
            check=True, capture_output=True, timeout=45).stdout

    def assert_tiles(self, raw, count, stage):
        frame_bytes = self.WIDTH * self.HEIGHT * 3
        self.assertEqual(len(raw), count * frame_bytes, stage + ' frame count/geometry')
        pixels = memoryview(raw)
        for frame in range(count):
            for row in range(4):
                for column in range(4):
                    # Central 16x16 interior, well away from codec/tile edges.
                    cx = column * self.TILE_WIDTH + self.TILE_WIDTH // 2
                    cy = row * self.TILE_HEIGHT + self.TILE_HEIGHT // 2
                    total = 0
                    for y in range(cy - 8, cy + 8):
                        start = frame * frame_bytes + (y * self.WIDTH + cx - 8) * 3
                        total += sum(pixels[start:start + 16 * 3])
                    mean = total / (16 * 16 * 3)
                    expected = self.level(frame, row, column)
                    self.assertLessEqual(
                        abs(mean - expected), self.MAX_ERROR,
                        f'{stage}: frame={frame} tile=({row},{column}) '
                        f'mean={mean:.3f} expected={expected}')

    def test_production_segment_then_obs_mpegts_preserves_spatial_tiles_and_order(self):
        timing = segment_timing(0)  # Actual production SegmentTiming fixture (124 frames).
        raw = self.raw_frames(timing.newly_emitted_frames)
        with tempfile.TemporaryDirectory(prefix='h3-double-encode-') as directory:
            first = Path(directory) / 'segment.mp4'
            second = Path(directory) / 'obs.ts'
            encode_rgb_segment(raw, first, timing, width=self.WIDTH, height=self.HEIGHT)
            decoded = self.decode(first)
            self.assert_tiles(decoded, timing.newly_emitted_frames, 'segment')

            tail = output_arguments(48124)
            self.assertTrue(tail[-1].startswith('udp://127.0.0.1:'))
            # Preserve every production output option; replace ONLY destination.
            tail = tail[:-1] + [str(second)]
            subprocess.run(
                ['ffmpeg', '-v', 'error', '-nostdin', '-f', 'rawvideo',
                 '-pixel_format', 'rgb24', '-video_size', f'{self.WIDTH}x{self.HEIGHT}',
                 '-framerate', '24', '-i', 'pipe:0'] + tail,
                input=decoded, check=True, capture_output=True, timeout=45)
            self.assert_tiles(self.decode(second), timing.newly_emitted_frames, 'OBS MPEG-TS')


if __name__ == '__main__':
    unittest.main()
