import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dataset1282_analysis.contracts import atomic_json, digest, read_json
from dataset1282_analysis.xu_geometry import CaseGeometry
from dataset1282_analysis.xu_roi import coverage, expand_box, measure_box, native_support, vote_box
from dataset1282_analysis.xu_roi_workflow import audit, localize, verified_book


def geometry(n=128):
    return CaseGeometry((n,)*3, (0.3,0.5,1.2), (0,1,2), ((0,n),)*3, (n,)*3)


class VotingTests(unittest.TestCase):
    def test_ten_percent_is_total_side_length_and_full_expansion_is_rejected(self):
        self.assertEqual(expand_box(((10,110),)*3,(128,)*3,10), [(5,115)]*3)
        self.assertEqual(expand_box(((0,100),)*3,(128,)*3,10), [(0,105)]*3)
        near = ((5,123),)*3
        self.assertTrue(measure_box(near,geometry())['accepted'])
        self.assertEqual(measure_box(expand_box(near,(128,)*3,10),geometry())['reason'],'whole_volume')

    def test_coverage_uses_reachable_centres_not_visual_enclosing_edges(self):
        geo = CaseGeometry((3,3,3),(1,1,1),(0,1,2),((0,3),)*3,(5,3,3))
        box = ((1,3),(0,3),(0,3))
        label = np.zeros((3,3,3),'u1')
        label[:,:,0] = 1
        self.assertEqual(coverage(label, {'vote':box},geo,[1])['vote'][1], 0)
        self.assertEqual(native_support(box,geo)[2], (1,2))
        self.assertEqual(measure_box(((1,2),(0,3),(0,3)),geo)['reason'], 'no_native_support')

    def test_support_matches_restore_on_random_asymmetric_boxes(self):
        rng = np.random.default_rng(74)
        geo = CaseGeometry((19,12,9),(.6,.7,1.1),(2,0,1),((2,18),(0,8),(1,10)),(23,5,14))
        for _ in range(50):
            box = []
            for n in geo.cache_shape:
                low = int(rng.integers(0,n))
                box.append((low,int(rng.integers(low+1,n+1))))
            got = np.zeros(geo.native_shape_xyz,'u1')
            got[tuple(slice(a,b) for a,b in native_support(box,geo))] = 1
            np.testing.assert_array_equal(got,geo.restore_label(np.ones((2,3,4),'u1'),box))

    def test_membership_conflicts_and_conservative_mapping(self):
        a = np.zeros((64,)*3, np.uint8)
        b = np.zeros((128,)*3, np.uint8)
        a[5,6,7] = 1
        b[10,12,15] = 6
        b[0,0,0] = 2
        vote = vote_box(a,b,geometry(129),0)
        self.assertEqual(vote['different_foreground'], 1)
        self.assertEqual(vote['same_foreground'], 0)
        self.assertEqual(vote['disputed_membership'], 8)
        self.assertEqual(vote['box'], [(10,12),(12,14),(15,17)])
        self.assertTrue(vote['quality']['accepted'])
        self.assertNotEqual(vote['legacy_box'], vote['box'])

    def test_disjoint_predictions_cannot_be_promoted(self):
        a = np.zeros((64,)*3, np.uint8)
        b = np.zeros((128,)*3, np.uint8)
        a[0,0,0] = b[-1,-1,-1] = 1
        vote = vote_box(a,b,geometry())
        self.assertIsNone(vote['box'])
        self.assertEqual(vote['quality']['reason'],'empty_vote')

    def test_whole_native_after_rounding_and_near_whole_are_rejected(self):
        geo = CaseGeometry((2,2,2),(1,1,1),(0,1,2),((0,2),)*3,(128,)*3)
        self.assertEqual(measure_box(((1,100),)*3,geo)['reason'], 'whole_volume')
        self.assertEqual(measure_box(((0,127),(0,128),(0,128)),geometry())['reason'],
                         'almost_whole_volume')
        self.assertTrue(measure_box(((0,90),(0,128),(0,128)),geometry())['accepted'])


class FixtureCache:
    def __init__(self):
        self.request_digest = 'test'
        self.supervision = {n:{'present_classes':[1]} for n in ('a','b')}
        self.predictions_only = True

    def case_ids(self,role):
        return ('a',) if role == 'train' else ('b',)

    def geometry(self,name):
        return geometry(32)

    def read_image(self,name,shape):
        a = np.zeros(shape, np.uint8)
        r = shape[0]//4
        a[r:3*r,r:3*r,r:3*r] = 1
        return a

    def read_native_label(self,name):
        if self.predictions_only:
            raise AssertionError('Localization accessed the reference')
        return self.read_image(name,(32,)*3)


class WorkflowTests(unittest.TestCase):
    def test_real_entrypoint_lifecycle_and_negative_gates(self):
        args = SimpleNamespace(supervision='marginal',profile='smoke',device='cpu',crop64='a',crop128='b',padding_percent=0)
        cache = FixtureCache()
        with TemporaryDirectory() as tmp, patch('dataset1282_analysis.xu_workflow.input_id', return_value='identity'):
            root = Path(tmp)
            with patch('dataset1282_analysis.xu_workflow.frozen_predictor',
                       side_effect=lambda p,**kw: (lambda x:x,'a'*64)):
                localize(cache,args,root)
            args.candidates = root/'candidates.json'
            self.assertFalse((root/'boxes.json').exists())
            cache.predictions_only = False
            audit(cache,args,root)
            self.assertEqual(len(verified_book(root/'boxes.json',cache,args)['boxes']), 2)
            envelope = read_json(args.candidates)
            envelope['payload']['cases']['b']['box'] = [[0,32]]*3
            envelope['payload']['cases']['b']['strict_box'] = [[0,32]]*3
            envelope['sha256'] = digest(envelope['payload'])
            atomic_json(args.candidates,envelope)
            failed = root/'rejected'
            with self.assertRaisesRegex(ValueError, 'rejected'):
                audit(cache,args,failed)
            self.assertFalse((failed/'boxes.json').exists())
            self.assertEqual(read_json(failed/'localization_qc.json')['failed_cases'], ['b'])
            # A book with updated hashes but an unsafe box must fail on the consumer side, too.
            book = read_json(root/'boxes.json')
            book['boxes']['b'] = [[0,32]]*3
            qc = read_json(root/'localization_qc.json')
            qc['book_digest'] = digest(book)
            atomic_json(root/'boxes.json',book)
            atomic_json(root/'localization_qc.json',qc)
            with self.assertRaisesRegex(ValueError, 'Invalid ROI'):
                verified_book(root/'boxes.json',cache,args)
