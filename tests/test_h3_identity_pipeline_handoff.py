import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from research import hf_identity_pipeline as pipeline


class PipelineHandoffTests(unittest.TestCase):
    def test_remote_fetch_bound_before_request_validation_offline_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'components').mkdir()
            (out / 'components/index.json').write_text('{}')
            (out / 'inventory.json').write_text('{}')
            request = dict(max_updates=100, rate_usd_hour=6.774192, spent_and_reserved_usd=0,
                territory_approval={}, model_repo='MiniMaxAI/MiniMax-H3', source_root='/source',
                output_repo='test/private', input_documents={'selection_path': {'path': '/selection'}},
                checkpoint_escrow=True, warm_start_remote=dict(repo_id='test/private', revision='a'*40,
                    prefix='identity/parent/escrow/checkpoint-0000025', metadata_sha256='b'*64))
            constants = NS(HF_HUB_OFFLINE=True)
            calls = []
            def fetch(**kw):
                self.assertFalse(constants.HF_HUB_OFFLINE)
                calls.append(kw)
                return dict(checkpoint='/input/checkpoint', metadata='/input/HANDOFF.json', metadata_sha256='b'*64)
            with patch.dict('sys.modules', {'huggingface_hub': NS(constants=constants)}), \
                 patch('research.h3_personal.identity_handoff_fetch.fetch_handoff', side_effect=fetch), \
                 patch.object(pipeline.trainer, 'validate_request') as validate:
                value = pipeline.train_request(request, out, 1200)
            self.assertTrue(constants.HF_HUB_OFFLINE)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['revision'], 'a'*40)
            self.assertEqual(len(calls[0]['expected_recipe_sha256']), 64)
            self.assertTrue(value['checkpoint_escrow'])
            self.assertEqual(value['warm_start']['metadata_sha256'], 'b'*64)
            validate.assert_called_once_with(value)
            self.assertEqual(json.loads((out / 'train-request.json').read_text()), value)


if __name__ == '__main__': unittest.main()
