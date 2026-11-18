"""Synthetic header/file plumbing only; not native image inference."""
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
import zlib
from research.h3_personal import ordered_references as refs
from research.hf_h3_direct_conversation import materialize_ordered_references


class Tests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup)
        self.root=Path(t.name).resolve();(self.root/'ordered-inputs').mkdir()
        self.output=self.root/'private';self.output.mkdir()
        self.config=dict(schema=refs.SCHEMA,references=[])
        for role,w,h in [('person',9,16),('surroundings',14,11)]:
            from test_h3_ordered_references import png_header
            data=png_header(w,h)
            p=self.root/'ordered-inputs'/(role+'.png');p.write_bytes(data)
            self.config['references'].append(dict(role=role,path=str(p),sha256=hashlib.sha256(data).hexdigest(),width=w,height=h))
        self.identity=refs.content_identity(self.config);self.sha=refs.content_sha256(self.config)

    def test_portable_parser_roundtrip_and_tamper(self):
        self.assertEqual(refs.validate_content_identity(self.identity),self.identity)
        self.assertEqual(refs.content_identity_sha256(self.identity),self.sha)
        for mutate in ('order','policy','extra','bool'):
            bad=copy.deepcopy(self.identity)
            if mutate=='order':bad['references'].reverse()
            if mutate=='policy':bad['preprocessing']['crop']='center'
            if mutate=='extra':bad['references'][0]['path']='/private/ignored.png'
            if mutate=='bool':bad['references'][0]['width']=True
            with self.assertRaises(ValueError):refs.validate_content_identity(bad)

    def test_immutable_direct_spec_binding_and_legacy_base_preserved(self):
        from test_h3_direct_conversation_spec import spec
        from test_h3_native_cancel_wiring import cancelled_config
        from research.hf_h3_direct_conversation import validate_spec
        value=spec();baseline=value['reference_sha256'];row=value['live_direct_session']
        row.update(tts_mode='native_voxcpm',native_tts=cancelled_config(),native_tts_cancel=True,
                   turn_interruptions=True,continuous_av=True,visual_intents=True,
                   ordered_reference_identity=self.identity,ordered_reference_sha256=self.sha)
        self.assertEqual(validate_spec(value),row)
        self.assertEqual(value['reference_sha256'],baseline)
        for change in ({'ordered_reference_sha256':'0'*64},{'visual_intents':False},
                       {'ordered_reference_identity':None}):
            bad=copy.deepcopy(value);bad['live_direct_session'].update(change)
            with self.assertRaises(ValueError):validate_spec(bad)
        bad=copy.deepcopy(value);del bad['live_direct_session']['ordered_reference_sha256']
        with self.assertRaises(ValueError):validate_spec(bad)
        bad=copy.deepcopy(value);bad['ordered_reference_identity']=self.identity
        with self.assertRaises(ValueError):validate_spec(bad)

    def test_copy_preserves_order_bytes_identity_and_private_permissions(self):
        result=materialize_ordered_references(self.root,self.identity,self.sha,self.output)
        self.assertEqual(refs.content_sha256(result),self.sha)
        for a,b in zip(self.config['references'],result['references']):
            self.assertEqual(Path(a['path']).read_bytes(),Path(b['path']).read_bytes())
            self.assertEqual(Path(b['path']).stat().st_mode&0o777,0o600)
        self.assertEqual(self.output.stat().st_mode&0o777,0o700)

    def test_changed_input_rejects_and_removes_partial_copy(self):
        (self.root/'ordered-inputs/surroundings.png').write_bytes(b'bad')
        with self.assertRaises(ValueError):
            materialize_ordered_references(self.root,self.identity,self.sha,self.output)
        self.assertEqual(list(self.output.iterdir()),[])

    def test_symlink_and_wrong_digest_rejected_before_native(self):
        path=self.root/'ordered-inputs/person.png';actual=self.root/'actual.png';path.rename(actual);path.symlink_to(actual)
        with self.assertRaises(ValueError):materialize_ordered_references(self.root,self.identity,self.sha,self.output)
        with self.assertRaises(ValueError):materialize_ordered_references(self.root,self.identity,'a'*64,self.output)


if __name__=='__main__':unittest.main()
