import json
from pathlib import Path
import tempfile
import unittest
from research import stream_modal_identity_progress as stream


class ProgressTests(unittest.TestCase):
    def test_explicit_read_only_command_and_finite_limits(self):
        cmd=stream.command('ta-example123',modal_cli='modal',interval=15,seconds=600)
        self.assertEqual(cmd[:6],['modal','container','exec','--no-pty','--','ta-example123'])
        self.assertNotIn('stop',cmd)
        self.assertNotIn('spawn',cmd)
        for container,interval,seconds in [('ap-example',15,600),('ta-example',0,600),('ta-example',15,3601)]:
            with self.assertRaises(ValueError):
                stream.command(container,modal_cli='modal',interval=interval,seconds=seconds)

    def test_display_unknown_totals_and_inventory_shrink(self):
        class Bar:
            def __init__(self, **kwargs): self.total=kwargs['total'];self.n=0
            def set_postfix(self, value, **kwargs): self.postfix=value
            def refresh(self): pass
            def close(self): pass
            def write(self, value): pass
        display=stream.ProgressDisplay(Bar)
        display.update(dict(pipeline_stage='prepare',cache=dict(bytes=100,files=3)))
        self.assertIsNone(display.bars['cache'].total)
        self.assertEqual(display.bars['cache'].n,100)
        display.update(dict(cache=dict(bytes=50,files=2),checkpoint_step=10,max_updates=200))
        self.assertEqual(display.bars['cache'].n,50)
        self.assertEqual(display.bars['checkpoint_step'].total,200)
        self.assertEqual(display.bars['checkpoint_step'].n,10)
        display.close()

    def test_snapshot_is_allowlisted_not_original_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def write(name,value):
                p=root/name.lstrip('/');p.parent.mkdir(parents=True,exist_ok=True)
                p.write_text(json.dumps(value))
            write('/tmp/h3-bootstrap-status.json',dict(stage='pipeline',failure_type=None,token='synthetic-secret'))
            write('/work/identity/pipeline-report.json',dict(stage='train',readiness=False,personal_text='never emit'))
            write('/work/identity/training/training-snapshot.json',dict(step=10,checkpoint='private-path'))
            write('/work/identity/training/latest-escrow.json',dict(parent_step=10))
            ns={}
            exec(stream.REMOTE_PROBE.split("interval=float(__import__('sys').argv[1])")[0],ns)
            ns['Path']=lambda value: root/str(value).lstrip('/')
            result=ns['snapshot']()
            self.assertEqual(result['pipeline_stage'],'train')
            self.assertEqual(result['checkpoint_step'],10)
            encoded=json.dumps(result)
            for excluded in ('synthetic-secret','personal_text','private-path'):
                self.assertNotIn(excluded,encoded)
            self.assertNotIn('loss',result)
            self.assertFalse(result['train_request_present'])


if __name__=='__main__':unittest.main()
