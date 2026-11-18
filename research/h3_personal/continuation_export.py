"""Lossless original recorded-audio mux and decode verification, local only."""
import hashlib
import json
from pathlib import Path
import subprocess


def recorded_wav_info(path):
    """FFmpeg supports PCM WAVE_FORMAT_EXTENSIBLE on Python3.11 too.

    Inspect only; never rewrite/transcode the original recording.
    """
    from fractions import Fraction
    from .geometry_preview import probe
    metadata=probe(path)
    streams=metadata['streams']
    if (len(streams)!=1 or streams[0].get('codec_type')!='audio' or
            'wav' not in metadata['format']['format_name'].split(',')):
        raise ValueError('single-stream PCM WAV required')
    a=streams[0]
    if a.get('codec_name')!='pcm_s24le' or int(a['bits_per_sample'])!=24:
        raise ValueError('original24bit little-endian PCM required')
    rate=int(a['sample_rate'])
    samples=Fraction(int(a['duration_ts']))*Fraction(a['time_base'])*rate
    if samples.denominator!=1 or samples<=0:
        raise ValueError('exact positive audio sample count required')
    return int(samples),rate,int(a['channels']),3


def pcm_sha(path):
    value=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(path),
        '-map','0:a:0','-c:a','pcm_s24le','-f','hash','-hash','sha256','-'],timeout=60).decode().strip()
    if not value.startswith('SHA256='):raise ValueError('PCM hash unavailable')
    return value.split('=',1)[1]


def mux(video,audio,output,*,expected_sha,samples=1364312,sample_rate=48000,frames=683):
    from .geometry_preview import digest,probe
    audio=Path(audio);output=Path(output)
    if digest(audio)!=expected_sha:raise ValueError('original WAV SHA differs')
    before=pcm_sha(audio)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(video),'-i',str(audio),
        '-map','0:v:0','-map','1:a:0','-map_metadata','-1','-map_chapters','-1',
        '-c:v','copy','-c:a','alac','-sample_fmt','s32p','-movflags','+faststart',str(output)],check=True,timeout=120)
    subprocess.run(['ffmpeg','-nostdin','-v','error','-xerror','-i',str(output),'-f','null','-'],check=True,timeout=120)
    metadata=probe(output);v=next(s for s in metadata['streams'] if s['codec_type']=='video')
    a=next(s for s in metadata['streams'] if s['codec_type']=='audio')
    if (int(v['nb_frames'])!=frames or a['codec_name']!='alac' or int(a['sample_rate'])!=sample_rate or
        int(a['channels'])!=1 or a['time_base']!=f'1/{sample_rate}' or int(a['duration_ts'])!=samples or
        pcm_sha(output)!=before or digest(audio)!=expected_sha):
        raise ValueError('full original PCM/frame/duration preservation failed')
    return dict(audio_source='user_recorded_original_lossless_mux',audio_codec='alac',
        audio_pcm_sha256=before,original_audio_samples=samples,sample_rate=sample_rate,
        frames=frames,full_ffmpeg_decode=True,original_wav_unchanged=True,
        full_original_pcm_preserved=True,no_tts=True,no_audio_segment_join=True,
        video_sha256=digest(output),video_bytes=output.stat().st_size)
