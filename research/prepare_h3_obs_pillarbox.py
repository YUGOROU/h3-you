"""Prepare a separate pillarboxed generated-only profile; never launch devices."""
import configparser
import copy
import json
from pathlib import Path

OLD = 'H3E2ELiveRelay20261025'
ID = 'H3E2EPillarbox20261025'
NAME = 'H3 E2E Pillarbox 20261025'
SCENE = 'Generated AV - HF relay - NO SENSORS'


def prepare(root):
    root = Path(root)
    profile = root/'basic/profiles'/ID
    target = root/'basic/scenes'/f'{ID}.json'
    if profile.exists() or target.exists():
        raise ValueError('fresh profile/collection required')
    data = json.loads((root/'basic/scenes'/f'{OLD}.json').read_text())
    sources = data['sources']
    if len(sources) != 2 or {s['id'] for s in sources} != {'scene', 'ffmpeg_source'}:
        raise ValueError('generated-only scene required')
    media = next(s for s in sources if s['id'] == 'ffmpeg_source')
    if (media['settings']['input'] != 'udp://127.0.0.1:48124?fifo_size=4096&overrun_nonfatal=0'
            or media['monitoring_type'] != 1 or media['mixers'] != 0):
        raise ValueError('approved AV route differs')
    data = copy.deepcopy(data)
    scene = next(s for s in data['sources'] if s['id'] == 'scene')
    items = scene['settings']['items']
    if len(items) != 1 or items[0]['source_uuid'] != media['uuid']:
        raise ValueError('single generated source required')
    # 252x448 source -> 607.5x1080, centered inside 1920x1080.
    items[0].update(pos={'x': 656.25, 'y': 0.0},
                    scale={'x': 1080/448, 'y': 1080/448},
                    rot=0.0, align=5, bounds_type=2, bounds_align=0,
                    bounds={'x': 607.5, 'y': 1080.0}, bounds_crop=False,
                    crop_left=0, crop_right=0, crop_top=0, crop_bottom=0)
    data.update(name=NAME, resolution={'x': 1920, 'y': 1080})
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(root/'basic/profiles'/OLD/'basic.ini')
    config['General']['Name'] = NAME
    for key, value in {'BaseCX':1920, 'BaseCY':1080, 'OutputCX':1920, 'OutputCY':1080}.items():
        config['Video'][key] = str(value)
    profile.mkdir(mode=0o700)
    with (profile/'basic.ini').open('x') as stream:
        config.write(stream, space_around_delimiters=False)
    target.write_text(json.dumps(data, indent=2)+'\n')
    target.chmod(0o600)
    (profile/'basic.ini').chmod(0o600)
    return {'profile': NAME, 'collection': NAME, 'scene': SCENE}
