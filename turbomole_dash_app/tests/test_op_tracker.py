"""Tests for backend.op_tracker."""

from backend import op_tracker


def setup_function():
    # Each test starts with an empty registry
    for jid in list(op_tracker.active()):
        op_tracker.mark_finished(int(jid))


def test_starts_empty():
    assert op_tracker.active() == {}
    assert not op_tracker.any_active()
    assert op_tracker.is_active(1) is None


def test_mark_and_finish():
    op_tracker.mark_started(7, "uploading")
    assert op_tracker.is_active(7) == "uploading"
    assert op_tracker.any_active() is True
    snap = op_tracker.active()
    assert 7 in snap
    assert snap[7]["kind"] == "uploading"

    op_tracker.mark_finished(7)
    assert op_tracker.is_active(7) is None
    assert op_tracker.active() == {}


def test_overwrite_with_new_kind():
    op_tracker.mark_started(3, "uploading")
    op_tracker.mark_started(3, "downloading")   # job 3 now downloading
    assert op_tracker.is_active(3) == "downloading"


def test_independent_jobs():
    op_tracker.mark_started(1, "uploading")
    op_tracker.mark_started(2, "downloading")
    snap = op_tracker.active()
    assert snap[1]["kind"] == "uploading"
    assert snap[2]["kind"] == "downloading"
    op_tracker.mark_finished(1)
    assert 1 not in op_tracker.active()
    assert 2 in op_tracker.active()


def test_finish_unknown_is_noop():
    op_tracker.mark_finished(999)  # no-op, no exception
    assert op_tracker.active() == {}


def test_rejects_bad_kind():
    import pytest
    with pytest.raises(AssertionError):
        op_tracker.mark_started(1, "weird")
