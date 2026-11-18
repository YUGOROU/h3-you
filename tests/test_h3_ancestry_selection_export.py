import unittest
from research.h3_personal.visual_ancestry_policy import validate_selection,CANDIDATE


def proof(seq):
    return dict(policy=CANDIDATE,sequence=seq,
                kind='unconditioned' if seq==0 else 'anchor' if seq%3==0 else 'latest',
                anchor_sha256=None if seq==0 else 'a'*64,
                selected_sha256=None if seq==0 else 'a'*64 if seq in (1,3,6) else 'b'*64,
                anchor_sequence=None if seq==0 else 1)


class SelectionTests(unittest.TestCase):
    def test_valid_finite_sequence(self):
        anchor=None
        for seq in range(8):
            row=validate_selection(proof(seq),seq,anchor)
            if seq==1:anchor=row['anchor_sha256']
            self.assertEqual(row,proof(seq))

    def test_tamper_fields_types_and_anchor(self):
        for seq in (0,1,2,3,6):
            for key,value in (('policy','latest'),('sequence',True),('anchor_sequence',True),
                              ('selected_sha256','private-text'),('extra','x')):
                row=proof(seq);row[key]=value
                with self.assertRaises(ValueError):validate_selection(row,seq,'a'*64)
        row=proof(3);row['selected_sha256']='b'*64
        with self.assertRaises(ValueError):validate_selection(row,3,'a'*64)
        with self.assertRaises(ValueError):validate_selection(proof(2),2,'c'*64)

if __name__=='__main__':unittest.main()
