import unittest
from pathlib import Path
from unittest.mock import patch
from research import hf_conversation_tts as tts


class VoxArchiveInstallTests(unittest.TestCase):
    def test_pinned_archive_does_not_require_git_or_mutate_h3(self):
        with patch.object(tts.subprocess,'run') as run:
            tts.install_vox(Path('/isolated/tts-env/bin/python'),Path('/isolated/constraints.txt'))
        run.assert_called_once()
        args=run.call_args.args[0];kw=run.call_args.kwargs
        self.assertEqual(args[0],'/isolated/tts-env/bin/python')
        self.assertIn('/isolated/constraints.txt',args)
        self.assertNotIn('git+',args[-1]);self.assertNotIn('git',args)
        self.assertIn('/tar.gz/'+tts.VOX_SOURCE+'#sha256='+tts.VOX_ARCHIVE_SHA256,args[-1])
        self.assertEqual(kw['env']['SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VOXCPM'],
                         '0.0.0+source.'+tts.VOX_SOURCE[:12])
        self.assertTrue(kw['check'])

    def test_install_failure_propagates_without_retry(self):
        with patch.object(tts.subprocess,'run',side_effect=RuntimeError('failed')) as run:
            with self.assertRaises(RuntimeError):tts.install_vox('/tts/python','/constraints')
        run.assert_called_once()
