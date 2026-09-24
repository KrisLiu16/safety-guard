"""Stage process-group supervisor, reused from the CPU-tested validation runner."""
import os
from pathlib import Path
import signal
import subprocess

def run_process_group(command, log_path, timeout, env=None, cwd=None):
    """Bound a whole stage, including its ordinary child/grandchild processes.

    The validation scripts spawn Python workers that inherit their process
    group. A new session keeps their forced timeout cleanup separate from this
    coordinator and from other jobs. SIGKILL is intentional: a child ignoring
    SIGTERM must not continue GPU work or write artifacts after timeout.
    """
    if os.name != 'posix':
        raise OSError('Validation stage process-group supervision requires POSIX')
    with Path(log_path).open('wb') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env, cwd=cwd)

        def stop_group():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # The complete group has already exited.
            # Reap the direct child after signalling the entire group. Never
            # stop after merely observing that the parent process has exited.
            process.wait()

        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            stop_group()
            error.process_group_id = process.pid
            error.process_group_termination = 'SIGKILL_sent_to_entire_group'
            raise
        except BaseException:
            stop_group()
            raise
    return subprocess.CompletedProcess(command, returncode)
