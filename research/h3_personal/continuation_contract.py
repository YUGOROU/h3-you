"""Explicit finite recorded-audio Turbo+CP inference scope, separate from matrix."""
AUDIO_SHA='def3cd0487dff827a09e99498c13cd1e07daee741ec258935763efe2dae0fad1'
SCOPE='h3-corrected-native-continuation-v1'


def validate(value):
    from .turbo_adapter import TURBO_SHA256
    expected=dict(schema=SCOPE,width=768,height=1344,total_frames=685,window_frames=277,
                  overlap_frames=22,video_shift=12,reference_policy='correct-hw-v2',
                  schedule_points=9,seed=314159)
    if not isinstance(value,dict) or set(value)!=set(expected)|{'turbo','audio'} or any(value[k]!=v for k,v in expected.items()):
        raise ValueError('exact native recorded-audio continuation scope required')
    turbo=value['turbo']
    if turbo!=dict(repo='lightx2v/Minimax-h3-Turbo',revision='3ec17a324ced54151364f24f8b5fb6bf7e26414f',
        path='/mounted-h3-turbo/minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors',
        file='minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors',bytes=1383677808,sha256=TURBO_SHA256):
        raise ValueError('immutable selected Turbo8 payload required')
    audio=value['audio']
    if (set(audio)!={'repo','revision','path','file','bytes','sha256'} or
        not isinstance(audio['repo'],str) or len(audio['repo'].split('/'))!=2 or
        audio['revision']!='fadd78ca615a06949fcad8ebab489b166d6fa6f1' or
        audio['path']!='/mounted-h3-native-audio/reference.wav' or audio['file']!='reference.wav' or
        audio['bytes']!=4093038 or audio['sha256']!=AUDIO_SHA):
        raise ValueError('immutable original recorded WAV pin required')
    return value
