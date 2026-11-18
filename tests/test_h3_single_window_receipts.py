"""Regression: one cold request must not read warm-only receipt files."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

class SingleWindowReceiptTests(unittest.TestCase):
    def test_no_warm_request_means_no_cache_receipt_read(self):
        tree=ast.parse(Path('research/run_h3_ref2va.py').read_text())
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cache_receipts' for t in n.targets))
        expr=compile(ast.Expression(assignment.value),'<cache-receipts>','eval')
        self.assertEqual(eval(expr,{'segments':1}),[])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'coverage').mkdir()
            for rank in (0,1):(root/'coverage'/f'cache-rank-{rank}.json').write_text('{"hits":1}')
            self.assertEqual(eval(expr,{'segments':2,'json':json,'a':SimpleNamespace(output=root)}),[{'hits':1},{'hits':1}])
    def test_packed_and_residency_warm_diagnostics_skipped_only_for_cold(self):
        tree=ast.parse(Path('research/run_h3_ref2va.py').read_text())
        for target in ('packed_receipts','residency_receipts'):
            clause=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and any(isinstance(v,ast.Name) and v.id=='segments' for v in ast.walk(n.test)) and any(isinstance(child,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in child.targets) for child in n.body))
            expr=compile(ast.Expression(clause.test),'<warm-only>','eval')
            args=SimpleNamespace(attention_mode='fa4-packed',runtime_policy='resident-visual-cached')
            self.assertFalse(eval(expr,{'segments':1,'a':args}));self.assertTrue(eval(expr,{'segments':2,'a':args}))

if __name__=='__main__':unittest.main()
