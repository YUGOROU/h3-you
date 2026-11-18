"""Application input boundary, not native EOS or semantic completion.

A bounded 16k PCM reader stops after >=10s and3s quiet following speech.
The ASR adapter must still commit and require genuine native full-input coverage.
"""
from array import array
import math


class SilenceBoundary:
    def __init__(self, stream, *, threshold=200, quiet_seconds=3, minimum_seconds=10):
        self.stream=stream;self.threshold=threshold;self.quiet=quiet_seconds
        self.minimum=minimum_seconds;self.frames=0;self.last_voice=None;self.done=False
        self.reason=None

    def read(self, size):
        if self.done:return b''
        block=self.stream.read(size)
        if not block:self.reason='source_eof';self.done=True;return b''
        if len(block)%2:raise ValueError('PCM16 alignment required')
        values=array('h');values.frombytes(block)
        self.frames+=len(values)
        rms=math.sqrt(sum(v*v for v in values)/len(values)) if values else 0
        seconds=self.frames/16000
        if rms>=self.threshold:self.last_voice=seconds
        if seconds>=self.minimum and self.last_voice is not None and seconds-self.last_voice>=self.quiet:
            self.done=True;self.reason='application_energy_silence'
        elif self.last_voice is None and seconds>=16:
            self.done=True;self.reason='no_voice_window'
        return block
