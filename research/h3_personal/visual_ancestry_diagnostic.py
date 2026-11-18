"""Finite generated-only ancestry comparison on one borrowed warm runtime."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import wave

SCHEMA = 'bounded-ancestry-comparison-v1'
ENV = 'H3_VISUAL_ANCESTRY_DIAGNOSTIC'
SECTION = dict(schema='h3-visual-ancestry-comparison-v1',
    arms=['epoch-idle-every3-v1', 'conversation-tail-every3-v1'], idle_before=6, idle_after=14)
LIMIT = 200 * 1024 * 1024


def digest(body):
    return hashlib.sha256(body).hexdigest()


def speech(root, contract):
    if type(contract) is not dict or set(contract) != {'filename', 'wav_sha256', 'pcm_sha256', 'samples', 'sample_rate', 'channels'}:
        raise ValueError('exact speech contract required')
    if (contract['filename'] != 'diagnostic-speech.wav' or contract['sample_rate'] != 48000
            or contract['channels'] != 1 or type(contract['samples']) is not int
            or not 0 < contract['samples'] <= 30*48000):
        raise ValueError('bounded PCM16 mono48k speech required')
    path = Path(root)/contract['filename']
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file() or path.stat().st_size > 3000000:
        raise ValueError('regular bounded speech file required')
    if digest(path.read_bytes()) != contract['wav_sha256']:
        raise ValueError('speech WAV hash mismatch')
    with wave.open(str(path), 'rb') as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes(), w.getcomptype()) != (1, 2, 48000, contract['samples'], 'NONE'):
            raise ValueError('speech WAV format mismatch')
        pcm = w.readframes(w.getnframes())
    if len(pcm) != contract['samples']*2 or digest(pcm) != contract['pcm_sha256']:
        raise ValueError('speech PCM mismatch')
    return pcm


def rank_evidence(coverage, manifest):
    """Read immediately, before retirement can relocate the receipt set."""
    c = manifest['streaming_audio']; seq = c['sequence']; rows = []
    if type(c['conversation_final']) is not bool or type(c['source_final']) is not bool:
        raise ValueError('strict streaming final booleans required')
    expected = 124+seq*102
    for rank in (0, 1):
        decode_path = Path(coverage)/f'h3-continuation-decode-rank-{rank}-window-0.json'
        if (any(q.is_symlink() for q in (decode_path, *decode_path.parents))
                or not decode_path.is_file() or decode_path.stat().st_size > 1048576):
            raise ValueError('bounded both-rank decode receipts required')
        decoded = json.loads(decode_path.read_bytes())
        if (decoded.get('session_id') != manifest['session_id']
                or type(decoded.get('worker_rank')) is not int or decoded['worker_rank'] != rank
                or type(decoded.get('committed_frames')) is not int or decoded['committed_frames'] != expected
                or type(decoded.get('emitted_frames')) is not int
                or decoded['emitted_frames'] != expected-(0 if c['conversation_final'] else 5)
                or type(decoded.get('final')) is not bool or decoded['final'] != c['conversation_final']):
            raise ValueError('decode rank/session/clock/final mismatch')
        p = Path(coverage)/f'streaming-rank-{rank}-window-{seq}.json'
        if any(q.is_symlink() for q in (p, *p.parents)) or not p.is_file() or p.stat().st_size > 1048576:
            raise ValueError('bounded both-rank receipts required')
        r = json.loads(p.read_bytes())
        if any(r['streaming'].get(k) != c.get(k) for k in ('sequence', 'role', 'logical_index', 'logical_source_id')):
            raise ValueError('rank sequence/role mismatch')
        guides = r['guide_digests']; canonical = r['canonical_sha256']
        if (type(guides) is not list or len(guides) != 3 or guides[2] != canonical
                or (seq > 0 and None in guides)):
            raise ValueError('canonical audio/visual evidence missing')
        for h in [canonical, *guides]:
            if h is not None and (type(h) is not str or len(h) != 64 or any(x not in '0123456789abcdef' for x in h)):
                raise ValueError('invalid evidence digest')
        evidence = r['visual_ancestry_evidence']
        if (type(evidence) is not dict or set(evidence) != {
                'policy', 'sequence', 'kind', 'anchor_sha256', 'selected_sha256', 'anchor_sequence'}):
            raise ValueError('exact actual visual ancestry evidence required')
        policy = manifest['visual_ancestry_diagnostic']['policy']
        anchored = seq > 0 and (seq % 3 == 0 if policy == SECTION['arms'][1] else
            c['role'] == 'listening' and c['logical_index'] >= 3 and c['logical_index'] % 3 == 0)
        if (evidence['policy'] != policy or type(evidence['sequence']) is not int
                or evidence['sequence'] != seq or evidence['selected_sha256'] != guides[0]
                or evidence['kind'] != ('unconditioned' if seq == 0 else 'anchor' if anchored else 'latest')):
            raise ValueError('selected guide/policy/kind mismatch')
        for key in ('anchor_sha256', 'selected_sha256'):
            h = evidence[key]
            if h is not None and (type(h) is not str or len(h) != 64 or any(x not in '0123456789abcdef' for x in h)):
                raise ValueError('invalid ancestry digest')
        if evidence['anchor_sha256'] is None:
            if evidence['anchor_sequence'] is not None or anchored:
                raise ValueError('missing immutable anchor')
        elif (type(evidence['anchor_sequence']) is not int or
                evidence['anchor_sequence'] != (1 if policy == SECTION['arms'][1] else
                                               seq-c['logical_index']+1)):
            raise ValueError('invalid anchor sequence')
        rows.append(dict(streaming={k:c[k] for k in ('sequence','role','logical_index','logical_source_id','source_final','conversation_final')},
            committed_frame=r['committed_frame'], emitted_frame=decoded['emitted_frames'],
            canonical_sha256=canonical, guide_digests=guides, visual_ancestry_evidence=evidence))
    if rows[0] != rows[1]:
        raise ValueError('both-rank ancestry disagreement')
    if type(rows[0]['committed_frame']) is not int or rows[0]['committed_frame'] != expected or rows[0]['emitted_frame'] != expected-(0 if c['conversation_final'] else 5):
        raise ValueError('rank clock mismatch')
    return rows[0]


def inventory(root, names):
    root = Path(root)
    if len(names) != len(set(names)):
        raise ValueError('duplicate inventory')
    total = 0
    for name in names:
        parts = Path(name).parts
        if (len(parts) != 2 or parts[0] not in ('arm-0', 'arm-1') or
                not (parts[1] in ('review.mov', 'evidence.json') or
                     __import__('re').fullmatch(r'(?:contact-[0-9]{3}\.png|native-[0-9]{3}\.mkv)', parts[1]))):
            raise ValueError('non-generated artifact name')
        p = root/name
        if any(q.is_symlink() for q in (p, *p.parents)) or not p.is_file():
            raise ValueError('regular generated artifact required')
        total += p.stat().st_size
        if total > LIMIT:
            raise ValueError('200 MiB aggregate artifact limit')
    return tuple(dict(path=n, bytes=(root/n).stat().st_size, sha256=digest((root/n).read_bytes())) for n in names)


def concat_manifest(segments, rows):
    """Production fragments carry global PTS; duration must exclude their origin."""
    if len(segments)!=len(rows):
        raise ValueError('one duration per generated segment required')
    previous=0
    parts=[]
    for p,row in zip(segments,rows):
        count=row['emitted_frame']-previous
        if count<=0:
            raise ValueError('strictly increasing segment clock required')
        parts.append("file 'segments/"+p.name+"'\nduration "+format(count/24,'.17g')+'\n')
        previous=row['emitted_frame']
    return ''.join(parts)


def run_diagnostic(runtime, root, *, pcm, deadline):
    from .live_native_backend import LiveNativeBackend, _export
    from .live_conversation import LiveConversation
    from .paired_av import LiveAVBinding
    from .conversation_conditioning import role_profile_digest
    from .streaming_pcm import PCMIngress, PCMStreamBinding, PCMChunk
    from .streaming_windows import ResamplingContext, NATIVE_TORCH, NATIVE_TORCHAUDIO, REFERENCE_SHA256
    from .streaming_trial import write_wav
    import numpy as np
    from PIL import Image
    if not math.isfinite(deadline) or deadline <= time.monotonic():
        raise ValueError('finite future deadline required')
    deadline = min(deadline, time.monotonic()+600)
    root = Path(root).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('private external output required')
    if os.environ.get(ENV) != SCHEMA or 'H3_IDLE_ANCHOR_DIAGNOSTIC' in os.environ:
        raise ValueError('exclusive ancestry environment required')
    root.mkdir(mode=0o700, exist_ok=False)
    names=[]; reports=[]
    def check():
        left=deadline-time.monotonic()
        if left <= 0: raise TimeoutError('comparison deadline')
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > LIMIT:
            raise ValueError('aggregate artifact limit')
        return left
    def command(args, **kw):
        return subprocess.run(['ffmpeg','-v','error','-xerror','-nostdin',*args],check=True,
                              timeout=check(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,**kw).stdout
    for arm_index, arm in enumerate(SECTION['arms']):
        directory=root/f'arm-{arm_index}';directory.mkdir(mode=0o700)
        rows=[]; hashes=[]; backend=driver=None; raw=directory/'temporary.pcm'
        def exporter(result,path,start,stop,index):
            check()
            pixels=np.stack([np.asarray(f) for f in result.frames])
            if pixels.dtype != np.uint8 or pixels.shape != (stop-start,448,256,3):
                raise ValueError('native RGB geometry mismatch')
            hashes.append(digest(pixels.tobytes()))
            if arm_index and index<6 and hashes[-1]!=reports[0]['native_rgb_sha256'][index]:
                raise ValueError('matched native RGB prefix differs')
            sheet=Image.new('RGB',(256*3,448))
            for slot,pick in enumerate((0,len(pixels)//2,len(pixels)-1)):
                sheet.paste(Image.fromarray(pixels[pick]),(256*slot,0))
            name=f'contact-{index:03d}.png'
            with (directory/name).open('xb') as out: sheet.save(out,format='PNG')
            names.append(f'arm-{arm_index}/{name}')
            c = rows[-1]['streaming']
            first_post_idle = (c['role'] == 'listening' and c['logical_index'] == 0
                               and any(r['streaming']['role'] == 'speech' for r in rows[:-1]))
            if (first_post_idle or c['conversation_final'] or
                    (c['role'] == 'speech' and c['source_final'])):
                native = f'native-{index:03d}.mkv'
                command(['-n','-f','rawvideo','-pix_fmt','rgb24','-s','256x448',
                         '-r','24','-i','pipe:0','-an','-c:v','ffv1','-level','3',
                         '-pix_fmt','gbrp',str(directory/native)], input=pixels.tobytes())
                names.append(f'arm-{arm_index}/{native}')
                check()
            video = _export(result,path,start,stop,index)
            check()
            return video
        def reader(coverage, manifest):
            from research.hf_h3_streaming_trial import read_evidence
            rows.append(rank_evidence(coverage,manifest))
            return read_evidence(coverage,manifest)
        try:
            backend=LiveNativeBackend(runtime,directory/'segments',exporter=exporter,evidence_reader=reader)
            profile=role_profile_digest(runtime.quality_profile,'speech')
            session=f'visual-ancestry-arm-{arm_index}'
            binding=LiveAVBinding(session,profile,1,max_segments=64,max_total_frames=6550)
            context=ResamplingContext('native-48k32k-prefix128',NATIVE_TORCH,NATIVE_TORCHAUDIO,REFERENCE_SHA256)
            driver=LiveConversation(directory/'driver',binding=binding,context=context,max_idle_windows=63,timeout=min(300,check()))
            def step(out):
                check();prepared=driver.prepare()
                if prepared is None: raise ValueError('unexpected diagnostic starvation')
                m=prepared.manifest
                m['visual_ancestry_diagnostic']=dict(schema=SCHEMA,policy=arm)
                packet=driver.complete_native(prepared,backend(m));out.write(packet.pcm)
            with raw.open('xb') as out:
                for _ in range(6): step(out)
                b=PCMStreamBinding(session,'diagnostic-speech',1,profile)
                stream=PCMIngress(b)
                for offset in range(0,len(pcm)//2,32000):
                    stream.append(PCMChunk(b,stream.snapshot()['chunks'],offset,pcm[offset*2:(offset+32000)*2]))
                stream.seal(total_samples=len(pcm)//2,sha256=digest(pcm));driver.attach(stream)
                while driver.inspect().status == 'ready': step(out)
                if driver.inspect().status != 'end': raise ValueError('speech EOS not completed')
                driver.retire_speech()
                for i in range(14):
                    if i == 13: driver.request_stop()
                    step(out)
            backend.retire()
            # Candidate refreshes on the GLOBAL clock, including speech.
            for row in rows:
                s=row['streaming'];seq=s['sequence'];e=row['visual_ancestry_evidence']
                anchor = None
                if arm_index and seq > 0:
                    anchor = rows[1]['guide_digests'][0]
                elif not arm_index and s['role'] == 'listening' and s['logical_index'] >= 1:
                    anchor = next(r['guide_digests'][0] for r in rows
                        if r['streaming']['logical_source_id'] == s['logical_source_id']
                        and r['streaming']['logical_index'] == 1)
                if e['anchor_sha256'] != anchor:
                    raise ValueError('immutable anchor digest mismatch')
                if e['kind'] == 'anchor' and row['guide_digests'][0] != anchor:
                    raise ValueError('refresh anchor digest mismatch')
            audio=raw.read_bytes();onset=(124+5*102)*4000
            if audio[onset:onset+len(pcm)] != pcm or any(audio[:onset]) or any(audio[onset+len(pcm):]):
                raise ValueError('original PCM offset/silence mismatch')
            wav=directory/'temporary.wav';write_wav(wav,audio)
            segments=sorted((directory/'segments').glob('segment-*.mp4'))
            if len(segments) != len(rows) or len(hashes) != len(rows) or len(audio) != rows[-1]['emitted_frame']*4000:
                raise ValueError('segment/frame/PCM count mismatch')
            listing=directory/'concat.txt';listing.write_text(concat_manifest(segments,rows))
            review=directory/'review.mov'
            command(['-n','-f','concat','-safe','0','-i',str(listing),'-i',str(wav),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','alac',str(review)])
            decoded=command(['-i',str(review),'-map','0:a:0','-f','s16le','pipe:1'])
            if len(decoded)!=len(audio) or digest(decoded)!=digest(audio): raise ValueError('decoded PCM mismatch')
            # Verify continuous external video time, not merely frame/sample counts.
            probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
                '-show_frames','-show_entries','frame=best_effort_timestamp_time',
                '-of','json',str(review)],check=True,timeout=check(),
                stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            pts=[float(f['best_effort_timestamp_time']) for f in json.loads(probe.stdout)['frames']]
            if len(pts)!=rows[-1]['emitted_frame'] or any(abs(t-i/24)>0.000002 for i,t in enumerate(pts)):
                raise ValueError('review video PTS drift')
            report=dict(policy=arm,windows=len(rows),native_rgb_sha256=hashes,receipts=rows,
                        pcm_sha256=digest(audio),samples=len(audio)//2,final=True)
            with (directory/'evidence.json').open('x') as f: json.dump(report,f,sort_keys=True,allow_nan=False)
            names.extend([f'arm-{arm_index}/review.mov',f'arm-{arm_index}/evidence.json'])
            reports.append(report)
            for p in segments: p.unlink()
            wav.unlink();listing.unlink()
        finally:
            raw.unlink(missing_ok=True)
            (directory/'temporary.wav').unlink(missing_ok=True)
            if driver is not None: driver.close()
            if backend is not None and not backend._retired:
                if backend._id is None: backend.close_unused()
                else: backend.abandon(reason='local-sink-failure')
    if reports[0]['native_rgb_sha256'][:6] != reports[1]['native_rgb_sha256'][:6]:
        raise ValueError('matched first six native RGB windows differ')
    entries=inventory(root,names)
    for entry in entries: (root/entry['path']).chmod(0o400)
    return dict(schema=SCHEMA,completed=True,arms=reports,files=entries)
