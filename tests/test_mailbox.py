"""Tests for mailbox.py - thread-safe message passing, QUIT stickiness,
last-writer-wins semantics, and event-driven waiting."""

import threading
import time
import pytest
from mailbox import Mailbox


# ---------------------------------------------------------------------------
# Basic post and check
# ---------------------------------------------------------------------------

class TestMailboxBasics:
    """Core post/check behavior."""

    def test_post_and_check(self):
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_PAUSE)
        assert mb.check() == Mailbox.TOGGLE_PAUSE

    def test_check_empty_returns_none(self):
        mb = Mailbox()
        assert mb.check() is None

    def test_check_clears_pending(self):
        """Reading a message consumes it."""
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_PAUSE)
        mb.check()
        assert mb.check() is None

    def test_post_all_message_types(self):
        """All three message types can be posted and read."""
        for msg in [Mailbox.TOGGLE_PAUSE, Mailbox.TOGGLE_ENABLE, Mailbox.QUIT]:
            mb = Mailbox()
            mb.post(msg)
            assert mb.check() == msg


# ---------------------------------------------------------------------------
# Last-writer-wins
# ---------------------------------------------------------------------------

class TestLastWriterWins:
    """Verify single-slot mailbox overwrites previous messages."""

    def test_second_post_overwrites_first(self):
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_PAUSE)
        mb.post(Mailbox.TOGGLE_ENABLE)
        assert mb.check() == Mailbox.TOGGLE_ENABLE

    def test_multiple_overwrites(self):
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_PAUSE)
        mb.post(Mailbox.TOGGLE_ENABLE)
        mb.post(Mailbox.TOGGLE_PAUSE)
        assert mb.check() == Mailbox.TOGGLE_PAUSE


# ---------------------------------------------------------------------------
# QUIT stickiness
# ---------------------------------------------------------------------------

class TestQuitSticky:
    """QUIT cannot be overwritten once posted."""

    def test_quit_cannot_be_overwritten(self):
        mb = Mailbox()
        mb.post(Mailbox.QUIT)
        mb.post(Mailbox.TOGGLE_PAUSE)
        assert mb.check() == Mailbox.QUIT

    def test_quit_overwritten_by_nothing(self):
        mb = Mailbox()
        mb.post(Mailbox.QUIT)
        mb.post(Mailbox.TOGGLE_ENABLE)
        mb.post(Mailbox.TOGGLE_PAUSE)
        assert mb.check() == Mailbox.QUIT

    def test_non_quit_overwritten_by_quit(self):
        """QUIT can overwrite a non-QUIT message."""
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_PAUSE)
        mb.post(Mailbox.QUIT)
        assert mb.check() == Mailbox.QUIT

    def test_quit_survives_after_check(self):
        """Once QUIT is consumed by check(), the mailbox is empty (not re-sticky)."""
        mb = Mailbox()
        mb.post(Mailbox.QUIT)
        assert mb.check() == Mailbox.QUIT
        # After consuming QUIT, mailbox is empty
        assert mb.check() is None


# ---------------------------------------------------------------------------
# wait() behavior
# ---------------------------------------------------------------------------

class TestMailboxWait:
    """Tests for event-driven wait()."""

    def test_wait_returns_posted_message(self):
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_PAUSE)
        result = mb.wait(timeout=0.1)
        assert result == Mailbox.TOGGLE_PAUSE

    def test_wait_returns_none_on_timeout(self):
        mb = Mailbox()
        start = time.monotonic()
        result = mb.wait(timeout=0.05)
        elapsed = time.monotonic() - start
        assert result is None
        assert elapsed >= 0.04  # waited at least close to the timeout

    def test_wait_wakes_on_post(self):
        """wait() returns before timeout when a message is posted."""
        mb = Mailbox()

        def delayed_post():
            time.sleep(0.05)
            mb.post(Mailbox.TOGGLE_ENABLE)

        t = threading.Thread(target=delayed_post)
        t.start()
        start = time.monotonic()
        result = mb.wait(timeout=2.0)
        elapsed = time.monotonic() - start
        t.join()

        assert result == Mailbox.TOGGLE_ENABLE
        assert elapsed < 1.0  # woke up well before 2s timeout

    def test_wait_consumes_message(self):
        """wait() consumes the message (subsequent check returns None)."""
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_PAUSE)
        mb.wait(timeout=0.1)
        assert mb.check() is None


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestMailboxThreadSafety:
    """Verify mailbox works correctly under concurrent access."""

    def test_concurrent_posts_no_crash(self):
        """Multiple threads posting simultaneously doesn't raise."""
        mb = Mailbox()
        errors = []

        def poster(msg, count):
            try:
                for _ in range(count):
                    mb.post(msg)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=poster, args=(Mailbox.TOGGLE_PAUSE, 100)),
            threading.Thread(target=poster, args=(Mailbox.TOGGLE_ENABLE, 100)),
            threading.Thread(target=poster, args=(Mailbox.QUIT, 50)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_quit_always_wins(self):
        """After concurrent posts including QUIT, the final state is QUIT."""
        mb = Mailbox()
        barrier = threading.Barrier(3)

        def post_toggle_pause():
            barrier.wait()
            for _ in range(50):
                mb.post(Mailbox.TOGGLE_PAUSE)

        def post_toggle_enable():
            barrier.wait()
            for _ in range(50):
                mb.post(Mailbox.TOGGLE_ENABLE)

        def post_quit():
            barrier.wait()
            mb.post(Mailbox.QUIT)
            # Post more after QUIT to verify stickiness
            for _ in range(50):
                mb.post(Mailbox.TOGGLE_PAUSE)

        threads = [
            threading.Thread(target=post_toggle_pause),
            threading.Thread(target=post_toggle_enable),
            threading.Thread(target=post_quit),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # QUIT was posted, so it must be the result (sticky)
        assert mb.check() == Mailbox.QUIT

    def test_post_during_wait_wakes_waiter(self):
        """A post from another thread wakes a waiting thread."""
        mb = Mailbox()
        results = []

        def waiter():
            result = mb.wait(timeout=2.0)
            results.append(result)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)  # let waiter enter wait()
        mb.post(Mailbox.TOGGLE_ENABLE)
        t.join(timeout=1.0)

        assert len(results) == 1
        assert results[0] == Mailbox.TOGGLE_ENABLE
