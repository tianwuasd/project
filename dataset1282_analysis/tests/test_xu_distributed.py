import copy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dataset1282_analysis import xu_distributed, xu_init, xu_training
from dataset1282_analysis.xu_distributed import BranchForward, split_sizes
from dataset1282_analysis.xu_resources import ResourceUnavailable


def cpu_replicas(model, chunks, devices):
    # Exercise split, ordered gather, loss and backward using independent copies of
    # the cases. Reusing the parameter tensors makes autograd sum their gradients.
    return torch.cat([model(chunk) for chunk in chunks], dim=0)


def emulated_group(count, batch_size=4):
    names = [f'cuda:{i}' for i in range(count)]
    with (patch.dict(os.environ, XU_BRANCH_DEVICES=json.dumps(names)),
          patch('dataset1282_analysis.xu_resources.require_applied_group') as gate):
        group = BranchForward(names[0], batch_size)
    gate.assert_called_once_with(names)
    return group


class RecordingAdam(torch.optim.Adam):
    def __init__(self, parameters):
        super().__init__(parameters, lr=2e-4)
        self.gradients = []

    def step(self):
        self.gradients.append([p.grad.detach().clone() for p in self.param_groups[0]['params']])
        return super().step()


class DistributedTests(unittest.TestCase):
    def test_incomplete_batches_keep_order_and_account_for_every_assigned_device(self):
        self.assertEqual(split_sizes(4, 2), (2, 2))
        self.assertEqual(split_sizes(4, 3), (2, 1, 1))
        self.assertEqual(split_sizes(4, 4), (1, 1, 1, 1))
        group = emulated_group(4)
        with patch.object(xu_distributed, '_replica_forward', side_effect=cpu_replicas):
            for n in (4, 2, 1, 3):
                x = torch.arange(n)[:, None]
                self.assertTrue(torch.equal(group.forward(torch.nn.Identity(), x), x))
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                group.forward(torch.nn.Identity(), torch.zeros(5, 1))
        self.assertEqual(group.stats['batches_by_layout'],
                         {'1,1,1,1': 1, '1,1,0,0': 1, '1,0,0,0': 1, '1,1,1,0': 1})
        self.assertEqual(list(group.stats['cases_by_device'].values()), [4, 3, 2, 1])
        seven = emulated_group(7, 4)
        self.assertEqual(seven.max_active, 4)
        self.assertEqual(emulated_group(3, 1).max_active, 1)
        with (patch.dict(os.environ, XU_BRANCH_DEVICES='["cuda:0","cuda:1"]'),
              patch('dataset1282_analysis.xu_resources.require_applied_group'),
              self.assertRaisesRegex(ValueError, 'exceeds the fixed branch batch')):
            BranchForward('cuda:0', 1, require_all=True)
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            split_sizes(0, 2)

    def test_group_requires_matching_explicit_distinct_applied_devices(self):
        for raw in ('[]', '["cpu"]', '["cuda"]', '["cuda:0","cuda:0"]',
                    '["cuda:1","cuda:0"]', '0', '[0]'):
            with self.subTest(raw=raw), patch.dict(os.environ, XU_BRANCH_DEVICES=raw):
                with self.assertRaises(ValueError):
                    BranchForward('cuda:0', 4)
        with (patch.dict(os.environ, XU_BRANCH_DEVICES='["cuda:0","cuda:1"]'),
              patch('dataset1282_analysis.xu_resources._applied_group', None)):
            with self.assertRaises(ResourceUnavailable):
                BranchForward('cuda:0', 4)

    def test_epoch_shards_match_whole_batch_gradients_adam_and_augmentation(self):
        ids = list('abcdef')
        missing = {'a': [], 'b': [4, 7], 'c': [9], 'd': [], 'e': [5, 8], 'f': []}
        raw = np.arange(64).reshape(4, 4, 4)
        images = {k: ((raw - 25 + i) / 16).astype('float32') for i, k in enumerate(ids)}
        labels = (raw % 4).astype('int8')
        labels[0, 0, 0] = -1
        cache = SimpleNamespace(read_image=lambda n, *args: images[n],
                                geometry=lambda n: SimpleNamespace(cache_shape=raw.shape))
        target = SimpleNamespace(read=lambda *args: (raw % 7).astype('int8'))
        book = {'boxes': {k: [] for k in ids}}
        old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, old_threads)
        torch.set_num_threads(1)
        for policy in ('complete', 'marginal', 'init'):
            is_init = policy == 'init'
            groups = missing if policy == 'marginal' else {n: [] for n in ids}
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(17)
                base = torch.nn.Conv3d(1, 7 if is_init else 11, 1)
            base.spec = SimpleNamespace(side=4, batch=4)
            for replicas in (1, 2, 3, 4):
                model = copy.deepcopy(base)
                opt = RecordingAdam(model.parameters())
                rng = torch.Generator().manual_seed(85)
                group = BranchForward('cpu', 4) if replicas == 1 else emulated_group(replicas)
                stats = {}
                module = xu_init if is_init else xu_training
                with (patch.object(module, 'BranchForward', return_value=group),
                      patch.object(xu_distributed, '_replica_forward', side_effect=cpu_replicas)):
                    if is_init:
                        loss = module._epoch(model, ids, cache, target, book, 'cpu', rng, opt, stats)
                    else:
                        loss = module._epoch(model, ids, groups, lambda n, _: (images[n], labels),
                                             'all64', 'cpu', rng, opt, stats)
                self.assertEqual(sum(stats['distribution']['cases_by_device'].values()), len(ids))
                state = [*model.parameters(), *[g for batch in opt.gradients for g in batch],
                         *[t for state in opt.state.values() for t in state.values()]]
                if replicas == 1:
                    baseline = loss, rng.get_state(), [t.detach().clone() for t in state]
                else:
                    with self.subTest(policy=policy, replicas=replicas):
                        self.assertAlmostEqual(loss, baseline[0], places=6)
                        self.assertTrue(torch.equal(rng.get_state(), baseline[1]))
                        for actual, expected in zip(state, baseline[2], strict=True):
                            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    def test_legacy_sources_preserve_prior_release_and_reject_changed_batch_contract(self):
        model = {'xu_models.py': 'fixed', 'training/marginal.py': 'same-loss'}
        current = dict(seed=1, source=dict(model, **{
            'xu_training.py': 'new', 'xu_supply.py': 'same', 'xu_distributed.py': 'new-helper'}))
        old = dict(seed=1, source=dict(model, **{
            'xu_training.py': '91418394469d70b9f5f1e3b16b2bc1309c40303c1c92ee12068343f77382cffc',
            'xu_supply.py': 'b66bf7ac0598dd54b33ba0930f26ab21c76daa72b08915eb6e85be1016642841'}))
        self.assertTrue(xu_training._compatible_contract(old, current))
        self.assertFalse(xu_training._compatible_contract(old, dict(current, batch_replicas=2)))
        self.assertFalse(xu_training._compatible_contract(dict(old, source=dict(old['source'], extra='x')), current))

    def test_read_only_replica_reuse_rejects_modified_weights_or_gradients(self):
        group = emulated_group(3)
        model = torch.nn.Linear(2, 1)
        with self.assertRaisesRegex(ValueError, 'evaluation-mode'):
            group.freeze(model)
        model.eval()
        with self.assertRaisesRegex(ValueError, 'frozen, versioned'):
            group.freeze(model)
        model.requires_grad_(False)
        with (patch.object(xu_distributed, '_check_model_device'),
              patch.object(xu_distributed.parallel, 'replicate',
                           side_effect=lambda module, devices, detach: [copy.deepcopy(module) for _ in devices]) as broadcast,
              patch.object(xu_distributed, '_replica_forward',
                           side_effect=lambda m, chunks, devices, frozen: torch.cat(
                               [net(x) for net, x in zip(frozen, chunks, strict=True)]))):
            group.freeze(model)
            with self.assertRaisesRegex(ValueError, 'no-grad'):
                group.forward(model, torch.zeros(3, 2))
            with torch.no_grad():
                for n in (4, 3, 2, 1):
                    x = torch.zeros(n, 2)
                    torch.testing.assert_close(group.forward(model, x), model(x))
                model.weight.add_(1)
                with self.assertRaisesRegex(ValueError, 'unchanged weights'):
                    group.forward(model, torch.zeros(3, 2))
            broadcast.assert_called_once()

    # Hardware integration is opt-in: the server must explicitly admit and cap every card.
    if os.environ.get('XU_TEST_CUDA_GROUP') == '1':
        def test_real_authorized_cuda_group_updates_with_global_loss(self):
            from dataset1282_analysis.xu_resources import apply_env_limit
            devices = json.loads(os.environ['XU_BRANCH_DEVICES'])
            self.assertGreaterEqual(len(devices), 2)
            self.assertLessEqual(len(devices), 4)
            primary = torch.device(devices[0])
            apply_env_limit(primary)
            group = BranchForward(primary, 4)
            model = torch.nn.Conv3d(1, 11, 3, padding=1).to(primary)
            reference = copy.deepcopy(model)
            x = torch.arange(4*64, device=primary, dtype=torch.float32).reshape(4, 1, 4, 4, 4)/64.
            y = torch.arange(4*64, device=primary).reshape(4, 4, 4, 4) % 3
            unknown = [[], [9], [8, 10], []]
            for m, scores in ((reference, reference(x)), (model, group.forward(model, x))):
                optimizer = RecordingAdam(m.parameters())
                loss = xu_training.branch_loss(scores, y, unknown)
                loss.backward()
                optimizer.step()
            self.assertTrue(all(group.stats['cases_by_device'].values()))
            for a, b in zip(reference.parameters(), model.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=1e-4, atol=3e-6)
