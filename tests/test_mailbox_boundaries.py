"""Tests for mailbox behavior at state boundaries - inter-state message
preservation, rapid posting semantics, and wakeup event clearing."""

import threading
import time
from mailbox import Mailbox


# ---------------------------------------------------------------------------
# Messages between state transitions
# ---------------------------------------------------------------------------

class TestMailboxBetweenStates:
    """Messages posted between state function return and next dispatch."""

    def test_message_posted_after_state_returns_is_preserved(self):
        """Message posted after state function returns but before next
        dispatch is preserved and visible to the next state."""
        mb = Mailbox()

        # Simulate: state_fn checks mailbox (empty), returns "listening"
        assert mb.check() is None

        # Between dispatches, GTK thread posts a message
        mb.post(Mailbox.TOGGLE_PAUSE)

        # Next state sees it
        assert mb.check() == Mailbox.TOGGLE_PAUSE

    def test_message_posted_between_dispatches_preserved(self):
        """Simulates state_fn returning, then post arriving, then next
        state checking - message must survive the gap."""
        mb = Mailbox()

        # State disabled returns "listening" (internally checked mailbox)
        result = mb.check()  # state consumed nothing
        assert result is None

        # Post arrives between state transitions
        mb.post(Mailbox.TOGGLE_ENABLE)

        # Next state (listening) sees it
        assert mb.check() == Mailbox.TOGGLE_ENABLE

    def test_multiple_posts_between_dispatches_last_wins(self):
        """Multiple posts between dispatches: last-write-wins, only final
        message seen by next state."""
        mb = Mailbox()

        mb.post(Mailbox.TOGGLE_PAUSE)
        mb.post(Mailbox.TOGGLE_ENABLE)
        mb.post(Mailbox.TOGGLE_PAUSE)

        # Only the last non-QUIT post survives
        assert mb.check() == Mailbox.TOGGLE_PAUSE


# ---------------------------------------------------------------------------
# Rapid posts (simulating GTK thread bursts)
# ---------------------------------------------------------------------------

class TestMailboxRapidPosts:
    """Rapid post sequences from the GTK thread."""

    def test_three_rapid_posts_last_non_quit_survives(self):
        """Three rapid posts from 'GTK thread': only last non-QUIT message survives."""
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_ENABLE)
        mb.post(Mailbox.TOGGLE_PAUSE)
        mb.post(Mailbox.TOGGLE_ENABLE)

        assert mb.check() == Mailbox.TOGGLE_ENABLE

    def test_quit_followed_by_toggle_quit_sticks(self):
        """QUIT followed by TOGGLE_PAUSE: QUIT sticks, TOGGLE_PAUSE ignored."""
        mb = Mailbox()
        mb.post(Mailbox.QUIT)
        mb.post(Mailbox.TOGGLE_PAUSE)

        assert mb.check() == Mailbox.QUIT

    def test_toggle_followed_by_quit_quit_wins(self):
        """TOGGLE_PAUSE followed by QUIT: QUIT wins."""
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_PAUSE)
        mb.post(Mailbox.QUIT)

        assert mb.check() == Mailbox.QUIT

    def test_rapid_posts_from_threads(self):
        """Multiple threads posting rapidly - no crashes, deterministic QUIT."""
        mb = Mailbox()
        barrier = threading.Barrier(2)

        def poster_a():
            barrier.wait()
            for _ in range(20):
                mb.post(Mailbox.TOGGLE_PAUSE)

        def poster_b():
            barrier.wait()
            mb.post(Mailbox.QUIT)

        ta = threading.Thread(target=poster_a)
        tb = threading.Thread(target=poster_b)
        ta.start()
        tb.start()
        ta.join()
        tb.join()

        # QUIT is sticky, must be the result
        assert mb.check() == Mailbox.QUIT


# ---------------------------------------------------------------------------
# check() clears wakeup event
# ---------------------------------------------------------------------------

class TestMailboxCheckClearsWakeup:
    """Verify check() resets the wakeup event properly."""

    def test_after_check_wait_blocks_until_next_post(self):
        """After check() consumes message, wait() blocks until next post
        (wakeup event cleared)."""
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_PAUSE)

        # Consume the message
        assert mb.check() == Mailbox.TOGGLE_PAUSE

        # Now wait should block (no pending message, wakeup cleared)
        start = time.monotonic()
        result = mb.wait(timeout=0.05)
        elapsed = time.monotonic() - start

        assert result is None
        assert elapsed >= 0.04  # Actually waited

    def test_check_empty_returns_none_wakeup_stays_cleared(self):
        """check() on empty mailbox: returns None, wakeup stays cleared."""
        mb = Mailbox()

        result = mb.check()
        assert result is None

        # Wakeup should still be cleared - wait() should block
        start = time.monotonic()
        result = mb.wait(timeout=0.05)
        elapsed = time.monotonic() - start

        assert result is None
        assert elapsed >= 0.04

    def test_wait_wakes_after_cleared_check(self):
        """After check() clears wakeup, a new post wakes wait()."""
        mb = Mailbox()
        mb.post(Mailbox.TOGGLE_PAUSE)
        mb.check()  # consume and clear

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
        assert elapsed < 1.0  # Woke up well before timeout
