"""Tests for text_processing.py - wake phrase stripping, punctuation, word
replacements, and break keyword detection."""

import pytest
from text_processing import (
    strip_wake_phrase,
    process_text,
    contains_break_keyword,
    remove_break_keyword,
)


# ---------------------------------------------------------------------------
# strip_wake_phrase
# ---------------------------------------------------------------------------

class TestStripWakePhrase:
    """Tests for removing the wake phrase from transcription output."""

    def test_strips_full_phrase(self, minimal_config):
        assert strip_wake_phrase("Hey Atlas how are you", minimal_config) == "how are you"

    def test_strips_partial_phrase(self, minimal_config):
        """Whisper sometimes only transcribes the last word of the wake phrase."""
        assert strip_wake_phrase("Atlas how are you", minimal_config) == "how are you"

    def test_case_insensitive(self, minimal_config):
        assert strip_wake_phrase("HEY ATLAS hello", minimal_config) == "hello"

    def test_strips_with_comma_after(self, minimal_config):
        """Whisper often inserts a comma after the wake phrase."""
        assert strip_wake_phrase("Hey Atlas, start typing", minimal_config) == "start typing"

    def test_strips_with_period_after(self, minimal_config):
        assert strip_wake_phrase("Hey Atlas. start typing", minimal_config) == "start typing"

    def test_no_wake_phrase_passthrough(self, minimal_config):
        """Text without wake phrase is returned unchanged."""
        assert strip_wake_phrase("just some text", minimal_config) == "just some text"

    def test_empty_wake_phrase_config(self, minimal_config):
        """Empty wake_phrase config disables stripping."""
        minimal_config['wake_phrase'] = ''
        assert strip_wake_phrase("Hey Atlas hello", minimal_config) == "Hey Atlas hello"

    def test_wake_phrase_only(self, minimal_config):
        """If transcription is just the wake phrase, result is empty."""
        result = strip_wake_phrase("Hey Atlas", minimal_config)
        assert result == ""

    def test_single_word_wake_phrase(self, minimal_config):
        """Single-word wake phrase uses simpler pattern."""
        minimal_config['wake_phrase'] = 'atlas'
        assert strip_wake_phrase("Atlas hello world", minimal_config) == "hello world"

    def test_wake_phrase_mid_sentence_not_stripped(self, minimal_config):
        """Wake phrase appearing mid-text is NOT stripped (only start)."""
        result = strip_wake_phrase("say hey atlas to start", minimal_config)
        assert "hey atlas" in result.lower() or "atlas" in result.lower()

    def test_strips_only_first_occurrence(self, minimal_config):
        """Only the leading wake phrase is removed."""
        result = strip_wake_phrase("Hey Atlas tell Atlas to stop", minimal_config)
        assert "Atlas" in result


# ---------------------------------------------------------------------------
# process_text - spoken punctuation
# ---------------------------------------------------------------------------

class TestProcessTextPunctuation:
    """Tests for spoken punctuation conversion."""

    def test_period_insertion(self, punctuation_config):
        result = process_text("hello world period", punctuation_config)
        assert result == "hello world."

    def test_comma_insertion(self, punctuation_config):
        result = process_text("hello comma world", punctuation_config)
        assert result == "hello, world"

    def test_question_mark(self, punctuation_config):
        result = process_text("how are you question mark", punctuation_config)
        assert result == "how are you?"

    def test_exclamation_point(self, punctuation_config):
        result = process_text("wow exclamation point", punctuation_config)
        assert result == "wow!"

    def test_new_line(self, punctuation_config):
        result = process_text("line one new line line two", punctuation_config)
        assert result == "line one\nline two"

    def test_new_paragraph(self, punctuation_config):
        result = process_text("paragraph one new paragraph paragraph two", punctuation_config)
        assert result == "paragraph one\n\nparagraph two"

    def test_parentheses(self, punctuation_config):
        result = process_text("hello open parenthesis aside close parenthesis world", punctuation_config)
        assert "(" in result
        assert "aside" in result
        assert ")" in result

    def test_colon(self, punctuation_config):
        result = process_text("note colon important", punctuation_config)
        assert result == "note: important"

    def test_semicolon(self, punctuation_config):
        result = process_text("first semicolon second", punctuation_config)
        assert result == "first; second"

    def test_ellipsis(self, punctuation_config):
        result = process_text("well dot dot dot maybe", punctuation_config)
        assert "..." in result
        assert "maybe" in result

    def test_multiple_punctuation_in_sequence(self, punctuation_config):
        result = process_text("hello comma how are you question mark", punctuation_config)
        assert "," in result
        assert "?" in result

    def test_multi_word_phrase_matched_before_single(self, punctuation_config):
        """'exclamation point' should match as a unit, not 'period' inside it."""
        result = process_text("wow exclamation point", punctuation_config)
        assert result == "wow!"
        assert "point" not in result

    def test_no_rules_passthrough(self, minimal_config):
        """With no punctuation rules, text passes through unchanged."""
        result = process_text("hello world", minimal_config)
        assert result == "hello world"


# ---------------------------------------------------------------------------
# process_text - word replacements
# ---------------------------------------------------------------------------

class TestProcessTextWordReplacements:
    """Tests for word replacement (Whisper mishearing correction)."""

    def test_basic_replacement(self, word_replacement_config):
        result = process_text("ask cloud about it", word_replacement_config)
        assert result == "ask Claude about it"

    def test_multiple_variants_same_target(self, word_replacement_config):
        """Different mishearings all map to the same correction."""
        assert "Claude" in process_text("ask cloud", word_replacement_config)
        assert "Claude" in process_text("ask clawed", word_replacement_config)

    def test_word_boundary_respected(self, word_replacement_config):
        """Replacement only applies to whole words, not substrings."""
        result = process_text("cloudy weather", word_replacement_config)
        assert "cloudy" in result

    def test_case_insensitive_matching(self, word_replacement_config):
        """re.IGNORECASE flag catches case variants."""
        result = process_text("ask CLOUD about it", word_replacement_config)
        assert "Claude" in result

    def test_sudo_replacement(self, word_replacement_config):
        result = process_text("run pseudo apt update", word_replacement_config)
        assert "sudo" in result

    def test_no_replacements_passthrough(self, minimal_config):
        result = process_text("no changes needed", minimal_config)
        assert result == "no changes needed"


# ---------------------------------------------------------------------------
# process_text - whitespace cleanup
# ---------------------------------------------------------------------------

class TestProcessTextWhitespace:
    """Tests for post-processing whitespace normalization."""

    def test_multiple_spaces_collapsed(self, minimal_config):
        result = process_text("hello    world", minimal_config)
        assert result == "hello world"

    def test_space_before_closing_punct_removed(self, minimal_config):
        result = process_text("hello .", minimal_config)
        assert result == "hello."

    def test_space_after_opening_punct_removed(self, minimal_config):
        result = process_text("( hello )", minimal_config)
        assert result == "(hello)"

    def test_leading_trailing_whitespace_stripped(self, minimal_config):
        result = process_text("  hello world  ", minimal_config)
        assert result == "hello world"


# ---------------------------------------------------------------------------
# contains_break_keyword
# ---------------------------------------------------------------------------

class TestContainsBreakKeyword:
    """Tests for session end phrase detection."""

    def test_exact_match(self, minimal_config):
        assert contains_break_keyword("break", minimal_config) is True

    def test_in_sentence(self, minimal_config):
        assert contains_break_keyword("ok break now", minimal_config) is True

    def test_case_insensitive(self, minimal_config):
        assert contains_break_keyword("BREAK", minimal_config) is True
        assert contains_break_keyword("Break", minimal_config) is True

    def test_variant_brake(self, minimal_config):
        """Whisper mishearing variant 'brake' is caught."""
        assert contains_break_keyword("brake", minimal_config) is True

    def test_variant_brick(self, minimal_config):
        """Whisper mishearing variant 'brick' is caught."""
        assert contains_break_keyword("brick", minimal_config) is True

    def test_word_boundary(self, minimal_config):
        """'break' inside another word does NOT trigger."""
        assert contains_break_keyword("breakfast", minimal_config) is False
        assert contains_break_keyword("unbreakable", minimal_config) is False

    def test_no_break_keyword(self, minimal_config):
        assert contains_break_keyword("hello world", minimal_config) is False

    def test_no_pattern_configured(self, minimal_config):
        """Missing pattern returns False."""
        minimal_config['end_phrase_pattern'] = None
        assert contains_break_keyword("break", minimal_config) is False

    def test_empty_text(self, minimal_config):
        assert contains_break_keyword("", minimal_config) is False


# ---------------------------------------------------------------------------
# remove_break_keyword
# ---------------------------------------------------------------------------

class TestRemoveBreakKeyword:
    """Tests for stripping the break keyword from text."""

    def test_removes_keyword(self, minimal_config):
        result = remove_break_keyword("hello break", minimal_config)
        assert result == "hello"

    def test_removes_keyword_with_trailing_punct(self, minimal_config):
        result = remove_break_keyword("hello break.", minimal_config)
        assert result == "hello"

    def test_removes_variant(self, minimal_config):
        result = remove_break_keyword("hello brake", minimal_config)
        assert result == "hello"

    def test_no_pattern_passthrough(self, minimal_config):
        minimal_config['end_phrase_pattern'] = None
        result = remove_break_keyword("hello break", minimal_config)
        assert result == "hello break"

    def test_keyword_only_returns_empty(self, minimal_config):
        result = remove_break_keyword("break", minimal_config)
        assert result == ""

    def test_keyword_mid_sentence(self, minimal_config):
        result = remove_break_keyword("say break now", minimal_config)
        assert "break" not in result.lower()
        assert "say" in result
