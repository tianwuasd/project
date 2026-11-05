from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class GlobalSlotSamplingTests(unittest.TestCase):
    def test_actual_native_augmentation_shards_reconstruct_exact_global_batch(self):
        import blosc2
        from chd_stunet.training.data import PatchSampler
        from chd_stunet.training.model import topology
        from chd_stunet.training.contracts import recipe
        from chd_stunet.preprocessing.stunet_s import bounded_locations
        from chd_stunet.preprocessing.annotations import make_annotation
        from chd_stunet.training.parallel_policy import slots_for
        scratch=Path(__file__).resolve().parents[1]/'outputs/tests-tmp';scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            base=Path(temporary)/'synthetic'
            seg=np.zeros((1,20,24,28),dtype=np.int8);seg[:,3:16,3:20,3:24]=1
            image=np.random.default_rng(81).normal(size=seg.shape).astype(np.float32)
            for suffix,array in (('.b2nd',image),('_seg.b2nd',seg)):
                blosc2.asarray(array,urlpath=str(base)+suffix,mode='w',cparams={'nthreads':1},dparams={'nthreads':1})
            key=tuple([-1]+list(range(8)))
            row=dict(base=str(base),shape=list(seg.shape),cohort_group='main',annotation=make_annotation([]),
                     properties=dict(class_locations=bounded_locations(seg,[1,list(key)])))
            rows={'synthetic':row};settings=recipe(True,dict(batch_size=4,patch_size=[16]*3))
            architecture=topology([1,1,1],[16]*3)
            seed=17313
            expected=PatchSampler(rows,list(rows),settings,architecture,True,[0,1,2,3])(seed)
            before=torch.get_rng_state().clone()
            for world in (2,3,4):
                shards=[PatchSampler(rows,list(rows),settings,architecture,True,slots_for(4,world,r))(seed)
                        for r in range(world)]
                np.testing.assert_array_equal(expected['data'],np.concatenate([s['data'] for s in shards]))
                for i,target in enumerate(expected['target']):
                    np.testing.assert_array_equal(target,np.concatenate([s['target'][i] for s in shards]))
                self.assertEqual(expected['cases'],[c for s in shards for c in s['cases']])
            torch.testing.assert_close(before,torch.get_rng_state(),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
