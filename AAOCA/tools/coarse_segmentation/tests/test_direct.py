import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaoca_seg.direct import gpu_indices, parser, positive


class DirectCommandTests(unittest.TestCase):
    def test_multiple_gpus_and_background_are_explicit_arguments(self):
        args = parser().parse_args(['run', '--manifest', 'input.csv', '--output-root', 'labels',
                                    '--gpus', '0,7', '--workers', '2', '--background'])
        self.assertEqual(args.gpus, [0, 7])
        self.assertEqual(args.workers, 2)
        self.assertTrue(args.background)
        self.assertEqual(args.min_free_gib, 8)
        self.assertEqual(args.gpu_reserve_gib, 2)

    def test_gpu_candidates_are_required_and_invalid_numbers_rejected(self):
        for value in ('', '0,0', '-1', '0,', 'a'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                gpu_indices(value)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            parser().parse_args(['run', '--manifest', 'input.csv', '--output-root', 'labels'])
        self.assertEqual(failure.exception.code, 2)

    def test_memory_threshold_rejects_nonfinite_or_zero_values(self):
        for value in ('nan', 'inf', '-1', '0'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                positive(value)

    def test_no_interactive_panel_command(self):
        for command in ('panel', 'serve'):
            with self.subTest(command=command), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args([command])

    def test_query_and_resume_keep_the_same_runtime(self):
        for command in ('status', 'logs', 'stop', 'resume'):
            args = parser().parse_args([command, '--runtime', 'private-runtime', '--job', 'a' * 32])
            self.assertEqual(args.runtime, 'private-runtime')
            self.assertEqual(args.job, 'a' * 32)

    def test_staged_controls_and_continuation_are_parsed(self):
        args = parser().parse_args(['run', '--manifest', 'input.csv', '--output-root', 'labels',
                                    '--gpus', '3,4', '--workers', '4', '--io-workers', '4',
                                    '--prepare-workers', '1', '--export-workers', '1', '--prefetch', '2',
                                    '--continue-from', 'latest'])
        self.assertEqual((args.prepare_workers, args.export_workers, args.prefetch), (1, 1, 2))
        self.assertEqual(args.continue_from, 'latest')


if __name__ == '__main__':
    unittest.main()
