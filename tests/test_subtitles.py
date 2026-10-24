from pathlib import Path

import pytest

from yakiflow.models import Cue, OutputMode
from yakiflow.subtitles import (
    ASS_HEADER,
    DEFAULT_PLAY_RES,
    AssEvent,
    SubtitleLayout,
    ass_header,
    alignment_problems,
    ass_problems,
    canonical_play_res,
    canonicalized_ass,
    cue_text_lines,
    event_problems,
    parse_ass,
    parse_timestamp,
    publish_outputs,
    render_ass,
    render_events,
    subtitle_layout,
    timestamp,
)


def _kinds(problems) -> list[str]:
    return [problem.kind for problem in problems]


# --- timestamps ---


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0:00:00.00"),
        (1.234, "0:00:01.23"),
        (1.235, "0:00:01.24"),
        (3661.5, "1:01:01.50"),
        (-0.5, "0:00:00.00"),
        (36000.0, "10:00:00.00"),
    ],
)
def test_timestamp_quantizes_to_centiseconds(seconds: float, expected: str) -> None:
    assert timestamp(seconds) == expected


@pytest.mark.parametrize("value", ["0:00:00.00", "1:02:03.45", "12:59:59.99"])
def test_timestamp_round_trip(value: str) -> None:
    parsed = parse_timestamp(value)
    assert parsed is not None
    assert timestamp(parsed) == value


def test_parse_timestamp_reads_single_fraction_digit_as_tenths() -> None:
    assert parse_timestamp("0:00:01.5") == 1.5


def test_parse_timestamp_rejects_garbage() -> None:
    assert parse_timestamp("not a time") is None
    assert parse_timestamp("00.00.00,0") is None


# --- render / parse round trip ---


def test_render_parse_round_trip_with_speakers_and_bilingual_lines() -> None:
    cues = [
        Cue("1", 0.0, 2.0, "Hello, world", "你好，世界", speaker="1"),
        Cue("2", 1.5, 3.5, "I disagree", "我不同意", speaker="2"),
        Cue("3", 4.0, 6.0, "narration", "旁白"),
    ]
    text = render_ass(cues, "bilingual")

    events, problems = parse_ass(text)

    assert problems == []
    assert [event.name for event in events] == ["1", "2", ""]
    assert events[0].text_lines == ("你好，世界", "Hello, world")
    # Commas in the Text field survive the field split.
    assert events[0].text_lines[1] == "Hello, world"
    assert (events[1].start, events[1].end) == (1.5, 3.5)
    assert event_problems(events) == []


def test_render_sanitizes_braces_and_folds_newlines() -> None:
    cue = Cue("1", 0.0, 2.0, "brace {override} here\nsecond", "第一\n第二")
    text = render_ass([cue], "bilingual")

    assert "{override}" not in text
    assert "｛override｝" in text
    events, _problems = parse_ass(text)
    assert events[0].text_lines == ("第一", "第二", "brace ｛override｝ here", "second")


def test_render_sanitizes_commas_in_speaker_names() -> None:
    cue = Cue("1", 0.0, 2.0, "hi", "嗨", speaker="a,b")
    events, problems = parse_ass(render_ass([cue], "bilingual"))
    assert problems == []
    assert events[0].name == "a b"


def test_cue_text_lines_modes() -> None:
    cue = Cue("1", 0.0, 1.0, "src", "tgt")
    assert cue_text_lines(cue, "source") == ("src",)
    assert cue_text_lines(cue, "translated") == ("tgt",)
    assert cue_text_lines(cue, "bilingual") == ("tgt", "src")
    untranslated = Cue("2", 0.0, 1.0, "src")
    assert cue_text_lines(untranslated, "translated") == ("src",)
    assert cue_text_lines(untranslated, "bilingual") == ("src",)


def test_parse_reports_unparsable_and_empty_dialogue_lines() -> None:
    text = (
        f"{ASS_HEADER}"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,ok\n"
        "Dialogue: 0,bogus,0:00:02.00,Default,,0,0,0,,broken time\n"
        "Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,\n"
        "Dialogue: too,few,fields\n"
    )
    events, problems = parse_ass(text)
    assert len(events) == 1
    assert sorted(_kinds(problems)) == ["empty_text", "unparsable", "unparsable"]


def test_dialogue_lines_outside_events_are_ignored() -> None:
    text = (
        "[Script Info]\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,not an event\n"
        "\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,real\n"
    )
    events, problems = parse_ass(text)
    assert problems == []
    assert len(events) == 1
    assert events[0].text_lines == ("real",)


# --- overlap matrix ---


def _event(start: float, end: float, name: str = "") -> AssEvent:
    return AssEvent(start, end, name, ("text",))


def test_different_named_speakers_may_overlap() -> None:
    events = [_event(0, 4, "1"), _event(2, 6, "2"), _event(3, 8, "3")]
    assert event_problems(events) == []


def test_same_speaker_overlap_is_a_defect() -> None:
    events = [_event(0, 4, "1"), _event(2, 6, "1")]
    assert _kinds(event_problems(events)) == ["overlap"]


def test_unnamed_overlapping_anything_is_a_defect() -> None:
    assert _kinds(event_problems([_event(0, 4), _event(2, 6)])) == ["overlap"]
    assert _kinds(event_problems([_event(0, 4, "1"), _event(2, 6)])) == ["overlap"]
    assert _kinds(event_problems([_event(0, 4), _event(2, 6, "1")])) == ["overlap"]


def test_non_adjacent_same_speaker_overlap_is_caught() -> None:
    # The interleaved speaker 2 hides the same-speaker overlap from any
    # neighbour-only comparison.
    events = [_event(0, 6, "1"), _event(1, 2, "2"), _event(3, 5, "1")]
    assert _kinds(event_problems(events)) == ["overlap"]


def test_touching_cues_do_not_overlap() -> None:
    events = [_event(0, 2, "1"), _event(2, 4, "1"), _event(4, 6)]
    assert event_problems(events) == []


def test_end_before_start_and_out_of_order_are_reported() -> None:
    events = [_event(5, 4, "1"), _event(3, 6, "2")]
    kinds = _kinds(event_problems(events))
    assert "end_before_start" in kinds
    assert "out_of_order" in kinds


def test_ass_problems_combines_parse_and_event_checks() -> None:
    text = (
        f"{ASS_HEADER}"
        "Dialogue: 0,0:00:02.00,0:00:04.00,Default,,0,0,0,,late\n"
        "Dialogue: 0,0:00:00.00,0:00:03.00,Default,,0,0,0,,early\n"
    )
    kinds = _kinds(ass_problems(text))
    assert "out_of_order" in kinds


# --- script resolution ---


def _style_fields(header: str) -> list[str]:
    line = next(
        line for line in header.splitlines() if line.startswith("Style:")
    )
    return line.split(":", 1)[1].split(",")


def test_ass_header_defaults_to_the_fallback_resolution() -> None:
    assert ass_header() == ASS_HEADER
    assert "PlayResX: 1920\nPlayResY: 1080\n" in ASS_HEADER


@pytest.mark.parametrize(
    ("play_res", "type_scale"),
    [
        # A vertical frame sizes the type off its width, an ultrawide one off
        # its height: whichever axis affords less.
        ((1080, 1920), 1080 / 1920),
        ((2560, 1080), 1.0),
        ((3840, 2160), 2.0),
    ],
)
def test_ass_header_scales_the_style_with_the_frame(
    play_res: tuple[int, int], type_scale: float
) -> None:
    header = ass_header(play_res)

    assert f"PlayResX: {play_res[0]}\nPlayResY: {play_res[1]}\n" in header
    fields = _style_fields(header)
    reference = _style_fields(ASS_HEADER)
    # Fontsize, Outline and Shadow keep the type the same size relative to the
    # picture; each margin is a fraction of the axis it is taken out of.
    scales = {
        2: type_scale,
        16: type_scale,
        17: type_scale,
        19: play_res[0] / 1920,
        20: play_res[0] / 1920,
        21: play_res[1] / 1080,
    }
    assert [float(fields[index]) for index in sorted(scales)] == [
        pytest.approx(float(reference[index]) * scale, abs=0.5)
        for index, scale in sorted(scales.items())
    ]
    # Everything else is the same style.
    assert [
        value for index, value in enumerate(fields) if index not in scales
    ] == [
        value for index, value in enumerate(reference) if index not in scales
    ]


def test_render_and_publish_author_against_the_given_resolution(
    tmp_path: Path,
) -> None:
    cues = [Cue("1", 0.0, 1.0, "hi", "嗨")]

    assert render_ass(cues, "bilingual", (3840, 2160)).startswith(
        ass_header((3840, 2160))
    )
    outputs = publish_outputs(
        tmp_path / "movie", cues, "source", play_res=(1280, 720)
    )
    text = outputs[0].read_text(encoding="utf-8")
    assert text.startswith(ass_header((1280, 720)))
    assert parse_ass(text)[1] == []


# --- canonicalization ---


def test_canonicalized_ass_rewrites_header_and_reorders_events() -> None:
    scrambled = (
        "[Script Info]\nScriptType: v4.00+\nTitle: custom edited header\n"
        "\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        "Dialogue: 0,0:00:05.0,0:00:06.00,Edited,later,0,0,0,,second\n"
        "Dialogue: 0,0:00:01.00,0:00:02.500,Default,,0,0,0,,first\n"
    )
    canonical = canonicalized_ass(scrambled)
    assert canonical is not None
    assert canonical.startswith(ASS_HEADER)
    assert "Title: custom edited header" not in canonical
    events, problems = parse_ass(canonical)
    assert problems == []
    assert [event.text_lines[0] for event in events] == ["first", "second"]
    assert canonical.index("first") < canonical.index("second")
    # Timestamps are normalized to H:MM:SS.cc.
    assert "0:00:05.00" in canonical
    # An already-canonical file needs no rewrite.
    assert canonicalized_ass(canonical) is None


def test_canonicalized_ass_rewrites_the_header_at_the_published_resolution() -> None:
    # A review that kept the file at the default resolution must be repaired
    # back to the resolution the job published at, not left mismatched.
    text = render_ass([Cue("1", 0.0, 1.0, "hi", "嗨")], "bilingual")
    canonical = canonicalized_ass(text, (1080, 1920))

    assert canonical is not None
    assert canonical.startswith(ass_header((1080, 1920)))
    assert canonicalized_ass(canonical, (1080, 1920)) is None


def test_canonicalized_ass_returns_none_for_unfixable_input() -> None:
    assert canonicalized_ass("this is not an ASS file") is None
    broken = (
        f"{ASS_HEADER}"
        "Dialogue: 0,bogus,0:00:02.00,Default,,0,0,0,,bad\n"
    )
    assert canonicalized_ass(broken) is None


def test_canonical_play_res_accepts_only_what_yakiflow_writes() -> None:
    cues = [Cue("1", 0.0, 1.0, "hi", "嗨", speaker="1")]
    written = render_ass(cues, "bilingual", (1080, 1920))

    assert canonical_play_res(written) == (1080, 1920)
    # Canonicalizing either of these would discard the user's styling.
    assert canonical_play_res(written.replace("sans-serif", "Noto Sans CJK SC")) is None
    assert canonical_play_res(written.replace(",嗨", r",{\i1}嗨")) is None
    assert canonical_play_res(ass_header((1080, 1920))) is None
    without_resolution = "".join(
        line for line in written.splitlines(keepends=True)
        if not line.startswith("PlayRes")
    )
    assert canonical_play_res(without_resolution) is None


def test_canonicalized_ass_preserves_legal_speaker_overlap() -> None:
    cues = [
        Cue("1", 0.0, 4.0, "a", "甲", speaker="1"),
        Cue("2", 2.0, 6.0, "b", "乙", speaker="2"),
    ]
    text = render_ass(cues, "bilingual")
    # Already canonical, overlap included.
    assert canonicalized_ass(text) is None
    events, _problems = parse_ass(text)
    assert event_problems(events) == []


# --- artifact alignment ---


def test_alignment_problems_reports_count_timing_and_name_drift() -> None:
    reference = [
        AssEvent(0.0, 1.0, "1", ("a",)),
        AssEvent(2.0, 3.0, "2", ("b",)),
    ]
    same = [
        AssEvent(0.0, 1.0, "1", ("x",)),
        AssEvent(2.0, 3.0, "2", ("y",)),
    ]
    assert alignment_problems([("ref", reference), ("same", same)]) == []

    fewer = [AssEvent(0.0, 1.0, "1", ("a",))]
    assert _kinds(
        alignment_problems([("ref", reference), ("fewer", fewer)])
    ) == ["artifact_mismatch"]

    drifted = [
        AssEvent(0.0, 1.0, "1", ("a",)),
        AssEvent(2.0, 3.5, "2", ("b",)),
    ]
    assert _kinds(
        alignment_problems([("ref", reference), ("drifted", drifted)])
    ) == ["artifact_mismatch"]

    renamed = [
        AssEvent(0.0, 1.0, "1", ("a",)),
        AssEvent(2.0, 3.0, "3", ("b",)),
    ]
    assert _kinds(
        alignment_problems([("ref", reference), ("renamed", renamed)])
    ) == ["artifact_mismatch"]


# --- publishing ---


def test_publish_outputs_names_ass_files_by_mode_and_language(
    tmp_path: Path,
) -> None:
    cues = [Cue("1", 0.0, 1.0, "hi", "嗨", speaker="1")]
    outputs = publish_outputs(
        tmp_path / "movie",
        cues,
        "all",
        source_language="en",
        target_language="zh-CN",
    )
    assert [path.name for path in outputs] == [
        "movie.source.ass",
        "movie.translated.ass",
        "movie.en-zh-cn.ass",
    ]
    for path in outputs:
        events, problems = parse_ass(path.read_text(encoding="utf-8"))
        assert problems == []
        assert events[0].name == "1"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("movie.source.ass", SubtitleLayout(OutputMode.SOURCE, "movie")),
        ("Movie.Translated.ASS", SubtitleLayout(OutputMode.TRANSLATED, "Movie")),
        (
            "movie.en-fr.ass",
            SubtitleLayout(OutputMode.BILINGUAL, "movie", "en", "fr"),
        ),
        (
            "movie.en-es-419.ass",
            SubtitleLayout(OutputMode.BILINGUAL, "movie", "en", "es-419"),
        ),
        # Both ja | zh-cn and ja-zh | cn are language-shaped.
        ("movie.ja-zh-cn.ass", SubtitleLayout(OutputMode.BILINGUAL, "movie")),
        ("movie.ass", None),
        ("movie.final.ass", None),
        ("movie.2024-01.ass", None),
        ("movie.source.srt", None),
        ("en-fr.ass", None),
    ],
)
def test_subtitle_layout_reads_published_names(
    name: str, expected: SubtitleLayout | None
) -> None:
    assert subtitle_layout(Path("subs") / name) == expected


def test_subtitle_layout_splits_an_ambiguous_pair_by_the_given_languages() -> None:
    path = Path("movie.ja-zh-cn.ass")

    assert subtitle_layout(path, target_language="zh_CN") == SubtitleLayout(
        OutputMode.BILINGUAL, "movie", "ja", "zh_CN"
    )
    assert subtitle_layout(path, source_language="ja") == SubtitleLayout(
        OutputMode.BILINGUAL, "movie", "ja", "zh-cn"
    )
    # A configured source language this file does not have does not stop
    # the target language from settling the split.
    assert subtitle_layout(
        path, source_language="en", target_language="zh-CN"
    ) == SubtitleLayout(OutputMode.BILINGUAL, "movie", "ja", "zh-CN")
    # "auto" names no language, and languages matching neither split leave
    # the pair as ambiguous as before.
    for hints in ({"source_language": "auto"}, {"target_language": "fr"}):
        assert subtitle_layout(path, **hints) == SubtitleLayout(
            OutputMode.BILINGUAL, "movie"
        )
    # An unambiguous pair is what the name says, whatever is configured.
    assert subtitle_layout(
        Path("movie.en-fr.ass"), source_language="ja", target_language="zh-CN"
    ) == SubtitleLayout(OutputMode.BILINGUAL, "movie", "en", "fr")


def test_publish_bilingual_requires_languages(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bilingual"):
        publish_outputs(tmp_path / "movie", [], "bilingual")


def test_render_events_is_stable_for_empty_input() -> None:
    assert render_events([]) == ASS_HEADER
    assert render_events([], DEFAULT_PLAY_RES) == ASS_HEADER
    events, problems = parse_ass(ASS_HEADER)
    assert events == [] and problems == []
