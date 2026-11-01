"""Timing must observe existing work without changing data or training steps."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '05_src'))
from training_timing import EpochTiming, TimedIterator, measure_step, summarize_steps


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class TrainingTimingTests(unittest.TestCase):
    def test_smoke_separates_batch_wait_from_step_and_preserves_sync_boundaries(self):
        clock = Clock()
        batch = object()
        def loader():
            clock.now += 7.
            yield batch
        def train(value):
            self.assertIs(value, batch)
            clock.now += .8
            return {'loss': 1.}
        sync = Mock()
        value, result, timing = measure_step(iter(loader()), train, sync, clock)
        self.assertIs(value, batch)
        self.assertEqual(result, {'loss': 1.})
        self.assertEqual(sync.call_count, 2)
        self.assertAlmostEqual(timing['batch_wait_seconds'], 7.)
        self.assertAlmostEqual(timing['step_seconds'], .8)
        self.assertAlmostEqual(timing['total_seconds'], 7.8)

    def test_summary_excludes_warmup_and_reports_actual_sample_count(self):
        records = [dict(batch_wait_seconds=t, step_seconds=1., total_seconds=t+1) for t in [99., 99., 1., 3., 2.]]
        summary = summarize_steps(records, warmup=2)
        self.assertEqual(summary['measured_steps'], 3)
        self.assertEqual(summary['warmup_steps'], 2)
        self.assertEqual(summary['total_seconds']['median'], 3.)
        self.assertEqual(summary['batch_wait_seconds']['sum'], 6.)
        with self.assertRaises(ValueError):
            summarize_steps(records[:2], warmup=2)

    def test_formal_observation_consumes_batches_once_and_stops_at_epoch_boundary(self):
        clock = Clock()
        timing = EpochTiming(clock=clock)
        batches = iter([object(), object()])
        wrapped = TimedIterator(batches, timing, 'train')
        # Reads during initialization do not count as epoch work.
        first = next(wrapped)
        timing.start(7)
        second = next(wrapped)
        self.assertIsNot(first, second)
        with timing.measure('train_step'):
            clock.now += 2.
        record = timing.finish()
        self.assertEqual(record['epoch'], 7)
        self.assertEqual(record['phases']['train_batch_wait']['count'], 1)
        self.assertEqual(record['phases']['train_step']['sum_seconds'], 2.)
        with self.assertRaises(StopIteration):
            next(wrapped)
        self.assertFalse(timing.active)

    def test_failed_operation_is_not_a_successful_timing_sample(self):
        timing = EpochTiming()
        timing.start(0)
        with self.assertRaises(RuntimeError):
            with timing.measure('train_step'):
                raise RuntimeError('failed update')
        self.assertNotIn('train_step', timing.finish()['phases'])

    def test_publication_is_append_only_and_io_failure_is_diagnostic(self):
        timing = EpochTiming()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'timing.jsonl'
            for epoch in [0, 1]:
                timing.start(epoch)
                self.assertIsNone(timing.publish(path, timing.finish()))
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record['epoch'] for record in records], [0, 1])
            with patch.object(Path, 'open', side_effect=OSError('disk unavailable')):
                self.assertIn('disk unavailable', timing.publish(path, records[-1]))


if __name__ == '__main__':
    unittest.main()
