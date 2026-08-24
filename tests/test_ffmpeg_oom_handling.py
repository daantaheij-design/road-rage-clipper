"""Tests for the OOM/resource-exhaustion-aware error handling in
ffmpeg_utils._run - the thing that used to just say "Command failed (-9)"
with no indication that -9 means the process was killed by the kernel."""

from __future__ import annotations

import signal

import pytest

from app.pipeline.ffmpeg_utils import FFmpegError, _describe_failure, _run


def test_normal_nonzero_exit_gets_plain_message():
    msg = _describe_failure(1, ["ffmpeg", "-i", "bad.mp4"], b"Invalid data found when processing input")
    assert "Command failed (1)" in msg
    assert "Invalid data found" in msg
    assert "OOM" not in msg
    assert "killed" not in msg


def test_sigkill_gets_a_clear_oom_message():
    msg = _describe_failure(-signal.SIGKILL, ["ffmpeg", "-i", "in.mp4", "out.mp4"], b"")
    assert "SIGKILL" in msg
    assert "OOM" in msg or "out of memory" in msg
    assert "killed" in msg


def test_sigbus_also_treated_as_likely_oom():
    msg = _describe_failure(-signal.SIGBUS, ["ffmpeg"], b"")
    assert "SIGBUS" in msg
    assert "out of memory" in msg


def test_other_signal_gets_generic_killed_message_not_oom_specific():
    msg = _describe_failure(-signal.SIGTERM, ["ffmpeg"], b"")
    assert "SIGTERM" in msg
    assert "killed" in msg
    # SIGTERM isn't a strong OOM signal - shouldn't claim it "almost always" means OOM.
    assert "almost always means the process ran out of memory" not in msg


async def test_run_raises_ffmpeg_error_with_oom_message_when_process_is_sigkilled():
    # Actually spawn and SIGKILL a process to exercise the real subprocess ->
    # returncode -9 -> FFmpegError path end-to-end, not just the pure
    # message-formatting function.
    with pytest.raises(FFmpegError) as exc_info:
        await _run(["bash", "-c", "kill -KILL $$"])
    message = str(exc_info.value)
    assert "SIGKILL" in message
    assert "out of memory" in message


async def test_run_normal_failure_is_not_misreported_as_oom():
    with pytest.raises(FFmpegError) as exc_info:
        await _run(["bash", "-c", "echo 'boom' >&2; exit 3"])
    message = str(exc_info.value)
    assert "Command failed (3)" in message
    assert "boom" in message
    assert "OOM" not in message
