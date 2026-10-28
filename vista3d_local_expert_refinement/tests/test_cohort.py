"""Catch leakage, accidental all-label fitting and duplicate-scan splits."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
from cohort import discover_cases, make_splits, validate_split, label_access_ids


def cases():
    result = []
    for group, prefix in [('A', 1000), ('B', 2000), ('G', 7000)]:
        for index in range(1, 21):
            case_id = f'Case{prefix+index}'
            result.append(dict(case_id=case_id, group=group+' ct_train', modality='CT',
                               image_sha256=hashlib.sha256(case_id.encode()).hexdigest(),
                               patient_id=None, status='quarantined' if case_id in ['Case2009','Case2017'] else 'eligible'))
    return result


class CohortTests(unittest.TestCase):
    def test_default_split_has_no_label_or_patient_overlap(self):
        split = make_splits(cases())['ct_holdG']
        self.assertEqual({k: len(split[k]) for k in ['train','val','selector','unlabeled','target']},
                         {'train':30,'val':8,'selector':0,'unlabeled':0,'target':20})
        self.assertEqual(len(label_access_ids(split)), 38)
        self.assertFalse(set(label_access_ids(split)) & set(split['unlabeled']+split['selector']+split['target']))
        self.assertTrue(all(x.startswith('Case7') for x in split['target']))
        self.assertEqual(make_splits(cases()),make_splits(list(reversed(cases()))))
        self.assertEqual(sum(x.startswith('Case1') for x in split['train']),16)
        self.assertEqual(sum(x.startswith('Case2') for x in split['train']),14)
        self.assertEqual(sum(x.startswith('Case1') for x in split['val']),4)
        self.assertNotIn('Case2009',sum([split[k] for k in ['train','val','selector','unlabeled','target']],[]))

    def test_duplicate_image_or_patient_requires_audited_grouping(self):
        for field in ['image_sha256','patient_id']:
            values=cases()
            values[0][field]=values[-1][field]='duplicate'
            with self.assertRaises(ValueError):make_splits(values)

    def test_tampered_split_cannot_train_on_hidden_or_target_cases(self):
        split=make_splits(cases())['ct_holdG']
        for forbidden in [split['target'][0],split['val'][0],'Case2009']:
            broken=copy.deepcopy(split)
            broken['train'][0]=forbidden
            with self.assertRaises(ValueError):validate_split(broken,cases())

    def test_discovery_does_not_read_label_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for group,prefix in [('A',1000),('B',2000),('G',7000)]:
                folder=root/(group+' ct_train');folder.mkdir()
                for n in range(1,21):
                    (folder/f'Case{prefix+n}_image.nii.gz').write_bytes(f'image{prefix+n}'.encode())
                    # Not even a valid NIfTI: discovery must not parse labels.
                    (folder/f'Case{prefix+n}_label.nii.gz').write_bytes(b'hidden label')
            found=discover_cases(root)
            self.assertEqual(len(found),60)
            self.assertEqual(sum(c['status']=='eligible' for c in found),58)
            self.assertTrue(all(c.get('label_sha256') is None for c in found))
            (root/'A ct_train/Case1001_label.nii.gz').unlink()
            with self.assertRaises(ValueError):discover_cases(root)


if __name__=='__main__':unittest.main()
