from __future__ import annotations

from app.pipeline.captions import Word, _group_words_into_phrases, build_ass_captions


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
