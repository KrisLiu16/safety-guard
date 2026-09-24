"""CPU regression: a timed-out parent cannot leave an artifact-writing child."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from run_final_validation import run_process_group


class ProcessGroupTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'The L20 validation worker uses POSIX process groups')
    def test_timeout_kills_sigterm_ignoring_child_and_next_stage_runs(self):
        child = """
import os, pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
(root / 'child_ready').write_text(str(os.getpid()))
print('child started', flush=True)
deadline = time.monotonic() + 5
with (root / 'child_heartbeat').open('ab', buffering=0) as stream:
    while time.monotonic() < deadline:
        stream.write(b'child still running\\n')
        time.sleep(.01)
"""
        parent = """
import os, pathlib, signal, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
signal.signal(signal.SIGTERM, signal.SIG_IGN)
subprocess.Popen([sys.executable, '-u', '-c', sys.argv[2], str(root)])
(root / 'parent_ready').write_text(str(os.getpid()))
print('parent started', flush=True)
deadline = time.monotonic() + 5
while time.monotonic() < deadline:
    time.sleep(.01)
"""
        with tempfile.TemporaryDirectory(prefix='validation-group-test-') as directory:
            root = Path(directory)
            with self.assertRaises(subprocess.TimeoutExpired) as raised:
                run_process_group([sys.executable, '-u', '-c', parent, str(root), child],
                                  root / 'timed_out.log', timeout=1.)
            self.assertTrue((root / 'parent_ready').is_file())
            self.assertTrue((root / 'child_ready').is_file())
            self.assertNotEqual(raised.exception.process_group_id, os.getpgrp())
            self.assertEqual(raised.exception.process_group_termination, 'SIGKILL_sent_to_entire_group')
            before = (root / 'child_heartbeat').read_bytes()
            self.assertGreater(len(before), 0)
            time.sleep(.25)
            self.assertEqual((root / 'child_heartbeat').read_bytes(), before,
                             'Child continued writing artifacts after the stage timeout')
            log = (root / 'timed_out.log').read_text()
            self.assertIn('parent started', log)
            self.assertIn('child started', log)
            following = run_process_group([sys.executable, '-c', "print('next stage completed')"],
                                          root / 'following.log', timeout=2.)
            self.assertEqual(following.returncode, 0)
            self.assertEqual((root / 'following.log').read_text().strip(), 'next stage completed')

    def test_nonzero_returncode_remains_explicit(self):
        with tempfile.TemporaryDirectory(prefix='validation-returncode-test-') as directory:
            result = run_process_group([sys.executable, '-c', "import sys; print('failure evidence'); sys.exit(7)"],
                                       Path(directory) / 'failed.log', timeout=2.)
            self.assertEqual(result.returncode, 7)
            self.assertIn('failure evidence', (Path(directory) / 'failed.log').read_text())


if __name__ == '__main__':
    unittest.main()
