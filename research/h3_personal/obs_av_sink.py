"""Opt-in continuous generated AV output, never OBS/device/routing setup.

Construction actually starts playback to localhost UDP and requires separate
operator approval. Import/command tests do not start it. Source input is exact
S16LE; MPEG-TS output uses lossy AAC (not byte-exact PCM at OBS/Discord).
"""
from .headless_av_sink import HeadlessAVSink


def output_arguments(port):
    if type(port) is not int or not 1024<=port<=65535:
        raise ValueError('explicit localhost UDP port required')
    return ['-c:v','libx264','-preset','ultrafast','-tune','zerolatency',
            '-pix_fmt','yuv420p','-threads','1','-g','24','-bf','0',
            '-c:a','aac','-b:a','192k','-mpegts_flags','+resend_headers+initial_discontinuity',
            '-muxdelay','0','-flush_packets','1','-f','mpegts',
            f'udp://127.0.0.1:{port}?pkt_size=1316']


class OBSAVSink(HeadlessAVSink):
    def __init__(self,root,binding,*,port,playback_approved=False,**kwargs):
        if playback_approved is not True:
            raise PermissionError('notify and obtain playback/routing approval before construction')
        self._obs_arguments=output_arguments(port)
        if 'paced' in kwargs and kwargs['paced'] is not True:
            raise ValueError('OBS continuous output must be paced')
        kwargs['paced']=True
        kwargs.setdefault('max_frames',124)
        super().__init__(root,binding,**kwargs)

    def _encoder_tail(self):
        return self._obs_arguments

    def _validate_output(self,deadline):
        # Parent has closed pipes and verified encoder exit. UDP provides no
        # receipt: never claim OBS acceptance or remote Discord delivery.
        return None
