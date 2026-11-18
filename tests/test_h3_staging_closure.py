"""Offline regression tests for shared deployment dependency closure."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research import launch_h3_conversation_scenario as staging


class StagingClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'source'
        self.stage = Path(self.temp.name) / 'stage'
        self.patch = patch.object(staging, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def module(self, name, text, *, staged=False):
        path = (self.stage if staged else self.root) / staging._module_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_package_from_import_closes_transitively(self):
        self.module('research.entry',
                    'from research.h3_personal import (continuation_diagnostic as diagnostic,)\n')
        self.module('research.h3_personal.continuation_diagnostic',
                    'from . import helper\nfrom .nested import child\n')
        self.module('research.h3_personal.helper', 'VALUE = 1\n')
        self.module('research.h3_personal.nested.child', 'from .. import helper\n')
        added = staging.closure(self.stage, ['research.entry'])
        self.assertEqual(set(added), {
            'research/entry.py',
            'research/h3_personal/continuation_diagnostic.py',
            'research/h3_personal/helper.py',
            'research/h3_personal/nested/child.py',
        })

    def test_existing_stage_is_traversed_without_overwrite(self):
        self.module('research.h3_personal.entry', 'raise RuntimeError("source version")\n')
        original = 'from . import middle\n'
        entry = self.module('research.h3_personal.entry', original, staged=True)
        self.module('research.h3_personal.middle',
                    'from . import entry, leaf\n', staged=True)
        self.module('research.h3_personal.leaf', 'VALUE = 2\n')
        added = staging.closure(self.stage, ['research.h3_personal.entry'] * 2)
        self.assertEqual(added, ['research/h3_personal/leaf.py'])
        self.assertEqual(entry.read_text(), original)
        self.assertEqual(staging.closure(self.stage, ['research.h3_personal.entry']), [])


if __name__ == '__main__':
    unittest.main()
