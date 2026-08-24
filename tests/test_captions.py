from __future__ import annotations

from app.pipeline.captions import (
    Word,
    _extend_narration_word_ends,
    _group_words_into_phrases,
    _is_emphasis_word,
    build_ass_captions,
)


def test_group_words_into_phrases_splits_on_gap():
    words = [
        Word("This", 0.0, 0.3),
        Word("driver", 0.3, 0.6),
        Word("cuts", 0.6, 0.9),
        Word("us", 5.0, 5.3),  # big gap -> new phrase
        Word("off", 5.3, 5.6),
    ]
    phrases = _group_words_into_phrases(words)
    assert len(phrases) == 2
    assert phrases[0][2] == "This driver cuts"
    assert phrases[1][2] == "us off"


def test_group_words_respects_max_words_per_phrase():
    words = [Word(f"w{i}", i * 0.2, i * 0.2 + 0.15) for i in range(12)]
    phrases = _group_words_into_phrases(words)
    for _, _, text in phrases:
        assert len(text.split()) <= 5


def test_build_ass_captions_writes_hook_and_transcript(tmp_path):
    words = [
        Word("he", 10.0, 10.2),
        Word("cuts", 10.2, 10.5),
        Word("us", 10.5, 10.7),
        Word("off", 10.7, 11.0),
    ]
    out = tmp_path / "captions.ass"
    build_ass_captions(
        output_path=out,
        clip_start=5.0,
        clip_end=30.0,
        transcript_words=words,
        narration_cues=[],
        hook_text="He never saw this coming.",
    )
    content = out.read_text()
    assert "He never saw this coming." in content
    assert "he cuts us off" in content
    assert "Style: Hook" in content
    assert "Style: Caption" in content


def test_narration_window_suppresses_transcript_captions(tmp_path):
    # transcript word sits inside the narration window (5s-8s relative) -
    # it should NOT appear as a separate Caption line.
    words = [Word("mumbled", 8.0, 9.0)]  # clip-relative: 3.0-4.0s
    out = tmp_path / "captions.ass"
    build_ass_captions(
        output_path=out,
        clip_start=5.0,
        clip_end=30.0,
        transcript_words=words,
        narration_cues=[(2.0, 5.0, "The narrator explains what happens.")],
        hook_text=None,
    )
    content = out.read_text()
    assert "The narrator explains what happens." in content
    assert "mumbled" not in content


def test_words_outside_clip_range_are_excluded(tmp_path):
    words = [Word("before", 0.0, 1.0), Word("after", 100.0, 101.0), Word("inside", 12.0, 12.5)]
    out = tmp_path / "captions.ass"
    build_ass_captions(
        output_path=out,
        clip_start=10.0,
        clip_end=20.0,
        transcript_words=words,
        narration_cues=[],
        hook_text=None,
    )
    content = out.read_text()
    assert "before" not in content
    assert "after" not in content
    assert "inside" in content


def test_is_emphasis_word_flags_exclamations_caps_and_long_words():
    assert _is_emphasis_word("watch!")
    assert _is_emphasis_word("really?")
    assert _is_emphasis_word("NOW")
    assert _is_emphasis_word("unbelievable")
    assert not _is_emphasis_word("the")
    assert not _is_emphasis_word("a")


def test_extend_narration_word_ends_fills_small_gaps_without_overlap():
    words = [Word("watch", 0.0, 0.3), Word("this", 0.5, 0.8), Word("now", 0.8, 1.0)]
    extended = _extend_narration_word_ends(words)
    # "watch" should stretch toward "this" start but never past it.
    assert extended[0].end <= words[1].start
    assert extended[0].end > words[0].end
    # Back-to-back words shouldn't be pushed into overlap.
    assert extended[1].end <= words[2].start


def test_build_ass_captions_renders_word_by_word_narration(tmp_path):
    narration_words = [
        Word("Watch", 0.0, 0.3),
        Word("this", 0.3, 0.6),
        Word("unbelievable", 0.6, 1.2),
        Word("moment!", 1.2, 1.6),
    ]
    out = tmp_path / "captions.ass"
    build_ass_captions(
        output_path=out,
        clip_start=0.0,
        clip_end=20.0,
        transcript_words=[],
        narration_cues=[(0.0, 1.6, "Watch this unbelievable moment!")],
        narration_words=narration_words,
        hook_text=None,
    )
    content = out.read_text()
    # Each word is its own event, not one line for the whole sentence.
    assert "Watch this unbelievable moment!" not in content
    assert content.count(",Narration,,0,0,0,,") == 2  # "Watch", "this" (not emphatic)
    assert ",NarrationEmphasis,,0,0,0,,unbelievable" in content
    assert ",NarrationEmphasis,,0,0,0,,moment!" in content


def test_build_ass_captions_falls_back_to_whole_line_without_word_timings(tmp_path):
    out = tmp_path / "captions.ass"
    build_ass_captions(
        output_path=out,
        clip_start=0.0,
        clip_end=20.0,
        transcript_words=[],
        narration_cues=[(0.0, 2.0, "No alignment data for this cue.")],
        narration_words=None,
        hook_text=None,
    )
    content = out.read_text()
    assert "No alignment data for this cue." in content


def test_narration_styles_positioned_mid_lower_not_bottom():
    from app.pipeline.captions import _HEADER

    for line in _HEADER.splitlines():
        if line.startswith("Style: Narration,") or line.startswith("Style: NarrationEmphasis,"):
            margin_v = int(line.split(",")[-2])
            # ~60-72% down a 1920px-tall frame means roughly 540-770px from
            # the bottom for a bottom-anchored (Alignment=2) style.
            assert 500 <= margin_v <= 800
