"""Offline admission for the Plan.md HTTPS conversation route.

No deployment, resume, capture or connection is performed here. In particular,
ASR WSS is an explicit migration exception, never called HTTPS body streaming.
Provider revision reconciliation must be supplied by the resource owner.
"""
from dataclasses import dataclass

from .voxcpm_endpoint_protocol import PINS


@dataclass(frozen=True)
class HTTPSConversationRoute:
    vox_revision: str
    asr_revision: str
    asr_wss_approved: bool = False

    def __post_init__(self):
        for revision in (self.vox_revision, self.asr_revision):
            if (not isinstance(revision, str) or len(revision) != 40
                    or any(c not in '0123456789abcdef' for c in revision)):
                raise ValueError('exact immutable deployment revision required')
        if type(self.asr_wss_approved) is not bool:
            raise ValueError('explicit ASR transport approval required')

    def inspect(self):
        return dict(tts_backend='h3-native-voxcpm-backend-v1',
                    tts_media='https-native-pcm',
                    output_media='https-continuous-paired-av',
                    output_control='independent-control-ack',
                    asr_media='wss', asr_https_body_streaming=False,
                    asr_transport_approved=self.asr_wss_approved,
                    rtc_fallback=False)

    def launch(self, directory, *, evidence, native_tts, turns=None,
               capture_comet=False, asr_parent=None, runner=None):
        """Compose the existing direct launcher after owner admission.

        This is an explicit live API, not an offline inspection operation.
        The caller retains finite resource ownership and operation approvals.
        No new Endpoint is created here. Evidence must be freshly reconciled
        by that owner; this function itself does not query provider state.
        """
        if type(evidence) is not dict:
            raise ValueError('owner deployment and READY evidence required')
        if type(capture_comet) is not bool:
            raise ValueError('explicit Comet capture choice required')
        if capture_comet != (asr_parent is not None):
            raise ValueError('capture and current ASR parent required together')
        if capture_comet and turns is not None:
            raise ValueError('typed turns cannot be mixed with capture')
        if capture_comet:
            import json
            from pathlib import Path
            parent = json.loads((Path(asr_parent) / 'intent.json').read_text())
            model = parent['model']
            if (model['revision'] != self.asr_revision
                    or model.get('env', {}).get('H3_ASR_MEDIA_TRANSPORT') is not None):
                raise ValueError('ASR parent must match reviewed legacy WSS revision')
        self.admit(native_tts=native_tts, output_transport='http-paired-v1',
                   continuous_av=True, **evidence)
        if runner is None:
            from research.launch_h3_direct_conversation import run
            runner = run
        if not callable(runner):
            raise ValueError('direct runner must be callable')
        return runner(directory, turns, capture_comet=capture_comet,
                      asr_parent=asr_parent, tts_mode='native_voxcpm',
                      native_tts=native_tts, conversation_anchor=True,
                      prefetch_receive=True, overlap_send=True,
                      http_keepalive=True, continuous_av=True,
                      media_transport='http-paired-v1')

    def admit(self, *, vox_revision, asr_revision, native_tts,
              vox_ready, expected_vox_ready, output_transport, continuous_av):
        """Validate BEFORE a caller opens input or submits a conversation.

        expected_vox_ready must come from the finite owner's current lease,
        not an old saved READY. This is a local guard, not provider attestation.
        Return the existing adapter class; do not construct it or send media.
        """
        if not self.asr_wss_approved:
            raise ValueError('ASR WSS migration exception requires user confirmation')
        if (vox_revision != self.vox_revision or asr_revision != self.asr_revision):
            raise ValueError('deployment revision mismatch')
        if (type(native_tts) is not dict
                or native_tts.get('schema') != 'h3-native-voxcpm-backend-v1'):
            raise ValueError('explicit native HTTPS Vox backend required; no fallback')
        if output_transport != 'http-paired-v1' or continuous_av is not True:
            raise ValueError('continuous HTTPS paired output required')
        if (type(vox_ready) is not dict or type(expected_vox_ready) is not dict
                or vox_ready.keys() != expected_vox_ready.keys()
                or any(type(vox_ready[k]) is not type(v) or vox_ready[k] != v
                       for k, v in expected_vox_ready.items())
                or any(vox_ready.get(k) != v for k, v in PINS.items())):
            raise ValueError('native Vox READY/lease mismatch')
        from .voxcpm_endpoint_client import RemoteVoxCPM
        return RemoteVoxCPM
