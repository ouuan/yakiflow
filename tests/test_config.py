import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from conftest import make_settings
from yakiflow.config import (
    SETTINGS_SCHEMA_VERSION,
    load_settings,
    validate_run_settings,
)
from yakiflow import elevenlabs as elevenlabs_module
from yakiflow.elevenlabs import elevenlabs_api_key, elevenlabs_api_key_source


MISSING = Path("/missing")


def _validatable(**overrides):
    values = {
        "source_language": "en",
        "target_language": "zh-CN",
        "agent": {"backend": "codex"},
    }
    values.update(overrides)
    return make_settings(values).resolved()


def test_config_priority(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    project = tmp_path / "project.toml"
    user.write_text(
        'target-language="fr"\noutput-mode="source"\n'
        '[agent.draft]\nworkers=2\n'
    )
    project.write_text('target-language="ja"\n[agent.draft]\nworkers=3\n')
    settings = load_settings(
        {"target_language": "zh-CN"},
        user_file=user,
        project_file=project,
        cli_config={"agent": {"draft": {"workers": 7}, "backend": "codex"}},
    )
    assert settings.target_language == "zh-CN"
    assert settings.agent.draft.workers == 7
    assert settings.output_mode == "source"
    assert settings.agent.draft.model == "gpt-5.6-terra"
    assert settings.agent.final.model == "gpt-5.6-sol"


def test_dedicated_flags_outrank_c_overrides(tmp_path: Path) -> None:
    settings = load_settings(
        {"target_language": "ja", "stream": True},
        user_file=MISSING,
        project_file=MISSING,
        cli_config={
            "target-language": "fr",
            "stream": False, "whisper": {"stream-chunk-seconds": 30},
        },
    )
    assert settings.target_language == "ja"
    assert settings.stream is True
    assert settings.whisper.stream_chunk_seconds == 30


def test_profiles_deep_merge_at_leaf_level(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    project = tmp_path / "project.toml"
    user.write_text(
        'target-language = "fr"\n'
        '[agent]\nbackend = "codex"\n'
        '[agent.draft]\nworkers = 2\nextra-options = ["--user-base"]\n'
        '[profiles.stream]\nstream = true\n'
        'agent.draft.workers = 3\n'
        'agent.draft.extra-options = ["-c", "service_tier=fast"]\n'
    )
    project.write_text(
        'target-language = "ja"\n'
        '[agent.draft]\nbatch-size = 10\n'
        '[profiles.stream]\ntarget-language = "zh-CN"\n'
        '[profiles.stream.agent.final]\nextra-options = ["--search"]\n'
    )

    settings = load_settings(
        user_file=user,
        project_file=project,
        profile="stream",
    )

    assert settings.stream is True
    assert settings.agent.backend == "codex"
    # The project base's batch-size survives the profile's sibling-key edits.
    assert settings.agent.draft.batch_size == 10
    assert settings.target_language == "zh-CN"
    assert settings.agent.draft.workers == 3
    assert settings.agent.draft.extra_options == ("-c", "service_tier=fast")
    assert settings.agent.final.extra_options == ("--search",)


def test_unselected_profiles_have_no_effect(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        'stream = false\n'
        '[profiles.streaming]\nstream = true\n'
    )

    settings = load_settings(user_file=MISSING, project_file=config)

    assert settings.stream is False


def test_unknown_profile_is_rejected(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[profiles.stream]\nstream = true\n')

    with pytest.raises(ValueError, match="unknown profile 'missing'"):
        load_settings(user_file=MISSING, project_file=config, profile="missing")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('profiles = "stream"\n', "profiles must be a table"),
        ('[profiles]\nstream = true\n', "profile 'stream' must be a table"),
        ('[profiles.stream]\ninherits = "base"\n', "cannot inherit"),
    ],
)
def test_malformed_profile_tables_are_rejected(
    tmp_path: Path, content: str, message: str
) -> None:
    config = tmp_path / "config.toml"
    config.write_text(content)

    with pytest.raises(ValueError, match=message):
        load_settings(user_file=MISSING, project_file=config)


def test_misspelled_configuration_key_is_reported(tmp_path: Path) -> None:
    project = tmp_path / "project.toml"
    project.write_text('targt-language="zh"\n')
    with pytest.raises(ValueError, match="unknown setting 'targt-language'"):
        load_settings(project_file=project, user_file=MISSING)


def test_misspelled_nested_key_is_reported_with_dotted_path(tmp_path: Path) -> None:
    project = tmp_path / "project.toml"
    project.write_text('[agent.draft]\nworker = 3\n')
    with pytest.raises(ValueError, match="unknown setting 'agent.draft.worker'"):
        load_settings(project_file=project, user_file=MISSING)


def test_relocated_flat_key_names_its_new_home(tmp_path: Path) -> None:
    project = tmp_path / "project.toml"
    project.write_text('translation_backend = "codex"\n')
    with pytest.raises(ValueError, match=r"moved to 'agent.backend'"):
        load_settings(project_file=project, user_file=MISSING)


def test_scalar_for_table_is_reported(tmp_path: Path) -> None:
    project = tmp_path / "project.toml"
    project.write_text('transcription = true\n')
    with pytest.raises(
        ValueError, match=r"'transcription' is a table of settings, not a single value"
    ):
        load_settings(project_file=project, user_file=MISSING)


def test_table_for_scalar_is_reported(tmp_path: Path) -> None:
    project = tmp_path / "project.toml"
    project.write_text('[whisper.stream-chunk-seconds]\nvalue = 15\n')
    with pytest.raises(
        ValueError, match=r"'whisper.stream-chunk-seconds' is a single setting, not a table"
    ):
        load_settings(project_file=project, user_file=MISSING)


def test_draft_only_keys_are_unknown_in_shared_and_final_tables(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared.toml"
    shared.write_text('[agent]\nworkers = 4\n')
    with pytest.raises(ValueError, match="unknown setting 'agent.workers'"):
        load_settings(project_file=shared, user_file=MISSING)

    final = tmp_path / "final.toml"
    final.write_text('[agent.final]\nbatch-size = 10\n')
    with pytest.raises(ValueError, match="unknown setting 'agent.final.batch-size'"):
        load_settings(project_file=final, user_file=MISSING)


def test_stage_resolution_prefers_stage_then_shared_then_default() -> None:
    settings = make_settings(
        agent={
            "backend": "codex",
            "effort": "medium",
            "draft": {"backend": "claude"},
        }
    ).resolved()

    # Draft overrides the backend, so its default model follows claude.
    assert settings.agent.draft.backend == "claude"
    assert settings.agent.draft.model == "sonnet"
    assert settings.agent.final.backend == "codex"
    assert settings.agent.final.model == "gpt-5.6-sol"
    # The shared effort outranks both stage defaults.
    assert settings.agent.draft.effort == "medium"
    assert settings.agent.final.effort == "medium"


def test_stage_effort_defaults_apply_when_nothing_is_set() -> None:
    settings = make_settings(agent={"backend": "codex"}).resolved()
    assert settings.agent.draft.effort == "low"
    assert settings.agent.final.effort == "high"
    assert settings.agent.draft.extra_options == ()
    assert settings.agent.final.extra_options == ()


def test_stage_extra_options_inherit_from_shared_table() -> None:
    settings = make_settings(
        agent={
            "backend": "codex",
            "extra-options": ["--shared"],
            "final": {"extra-options": ["--final-only"]},
        }
    ).resolved()
    assert settings.agent.draft.extra_options == ("--shared",)
    assert settings.agent.final.extra_options == ("--final-only",)


def test_only_the_batch_elevenlabs_backend_reports_diarization() -> None:
    assert make_settings(transcription={"backend": "elevenlabs"}).diarization_enabled
    assert not make_settings(
        transcription={"backend": "elevenlabs"}, elevenlabs={"diarize": False}
    ).diarization_enabled
    assert not make_settings(
        transcription={"backend": "elevenlabs-realtime"}
    ).diarization_enabled
    assert not make_settings().diarization_enabled


def test_resolved_is_idempotent() -> None:
    settings = _validatable()
    assert settings.resolved() == settings


def test_extra_options_require_arrays_of_strings(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[agent.draft]\nextra-options = ["--search", 1]\n')

    with pytest.raises(
        ValueError, match="agent.draft.extra-options must be an array of strings"
    ):
        load_settings(user_file=MISSING, project_file=config)


def test_yt_dlp_options_require_arrays_of_strings(tmp_path: Path) -> None:
    config = tmp_path / "project.toml"
    config.write_text('[commands]\nyt-dlp-options = ["--cookies-from-browser", 1]\n')

    with pytest.raises(
        ValueError, match="commands.yt-dlp-options must be an array of strings"
    ):
        load_settings(user_file=MISSING, project_file=config)


def test_path_settings_are_canonicalized_when_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings(
        cli_config={
            "whisper": {"model": "models/whisper.bin", "vad-model": "models/vad.bin"},
            "download-dir": "downloads",
            "output-dir": "subtitles",
            "memory": "state/memory.md",
            "context-files": ["references/chat.xml", "references/notes.txt"],
            "work-dir": "work",
        },
        user_file=MISSING,
        project_file=MISSING,
    )

    assert settings.whisper.model == tmp_path / "models/whisper.bin"
    assert settings.whisper.vad_model == tmp_path / "models/vad.bin"
    assert settings.download_dir == tmp_path / "downloads"
    assert settings.output_dir == tmp_path / "subtitles"
    assert settings.memory == tmp_path / "state/memory.md"
    assert settings.context_files == (
        tmp_path / "references/chat.xml",
        tmp_path / "references/notes.txt",
    )
    assert settings.work_dir == tmp_path / "work"


def test_vad_can_be_disabled_without_unsetting_the_model(tmp_path: Path) -> None:
    project_file = tmp_path / "yakiflow.toml"
    project_file.write_text('[whisper]\nvad-model = "/models/vad.bin"\n')

    enabled = load_settings(user_file=MISSING, project_file=project_file)
    disabled = load_settings(
        cli_config={"whisper": {"vad": False}},
        user_file=MISSING,
        project_file=project_file,
    )

    assert enabled.whisper.vad_model == Path("/models/vad.bin")
    assert disabled.whisper.vad_model is None


def test_stored_settings_round_trip(tmp_path: Path) -> None:
    original = _validatable(
        context_files=[str(tmp_path / "notes.txt")],
        elevenlabs={"api-key": "sk-stored"},
        stream=True,
    )
    stored = asdict(original)

    restored = load_settings(
        stored=stored,
        user_file=MISSING,
        project_file=MISSING,
    )

    assert restored == original


def test_stored_internal_fields_round_trip_but_config_files_cannot_set_them(
    tmp_path: Path,
) -> None:
    original = replace(
        _validatable(),
        elevenlabs=replace(
            _validatable().elevenlabs, api_key="sk", api_key_from_cli=True
        ),
    )
    restored = load_settings(
        stored=asdict(original), user_file=MISSING, project_file=MISSING
    )
    assert restored.elevenlabs.api_key_from_cli is True

    config = tmp_path / "config.toml"
    config.write_text('[elevenlabs]\napi-key-from-cli = true\n')
    with pytest.raises(
        ValueError, match="unknown setting 'elevenlabs.api-key-from-cli'"
    ):
        load_settings(user_file=MISSING, project_file=config)


# --- validation ---


def test_required_languages_and_agent_backend() -> None:
    with pytest.raises(ValueError, match="target-language"):
        validate_run_settings(make_settings().resolved())
    with pytest.raises(ValueError, match="source-language"):
        validate_run_settings(
            make_settings(
                target_language="zh-CN", agent={"backend": "codex"}
            ).resolved()
        )
    with pytest.raises(ValueError, match="agent.backend is required"):
        validate_run_settings(
            make_settings(source_language="en", target_language="zh-CN").resolved()
        )


def test_auto_is_an_explicit_source_language() -> None:
    validate_run_settings(_validatable(source_language="auto"))


def test_transcription_backend_is_validated() -> None:
    with pytest.raises(ValueError, match="transcription.backend"):
        validate_run_settings(
            _validatable(transcription={"backend": "whisperx"})
        )


@pytest.mark.parametrize(
    "backend", ["whisper-cli", "whisper-server", "elevenlabs", "elevenlabs-stream"]
)
@pytest.mark.parametrize("stream_enabled", [False, True])
def test_every_backend_may_stream(
    backend: str,
    stream_enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    validate_run_settings(
        _validatable(
            transcription={"backend": backend},
            stream=stream_enabled,
        )
    )


def test_alignment_backend_defaults_follow_transcription_backend() -> None:
    assert _validatable().alignment.backend == "vad"
    assert (
        _validatable(transcription={"backend": "whisper-server"}).alignment.backend
        == "vad"
    )
    assert (
        _validatable(transcription={"backend": "elevenlabs"}).alignment.backend
        == "none"
    )
    assert (
        _validatable(
            transcription={"backend": "elevenlabs-stream"}
        ).alignment.backend
        == "none"
    )


def test_vad_alignment_is_rejected_for_word_level_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    with pytest.raises(ValueError, match="alignment.backend 'vad'"):
        validate_run_settings(
            _validatable(
                transcription={"backend": "elevenlabs"},
                alignment={"backend": "vad"},
            )
        )


def test_diarized_elevenlabs_accepts_whisperx_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Forced alignment leaves crosstalk cues at their ASR times, so it no
    # longer has to be refused outright when diarization is on.
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test")
    validate_run_settings(
        _validatable(
            transcription={"backend": "elevenlabs"},
            alignment={"backend": "whisperx"},
        )
    )
    validate_run_settings(
        _validatable(
            transcription={"backend": "elevenlabs"},
            elevenlabs={"diarize": False},
            alignment={"backend": "whisperx"},
        )
    )


def test_alignment_none_is_accepted_for_whisper_backends() -> None:
    validate_run_settings(_validatable(alignment={"backend": "none"}))


def test_whisper_server_url_must_be_http(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="whisper.server-url"):
        validate_run_settings(
            _validatable(whisper={"server-url": "ftp://example.com"})
        )
    validate_run_settings(
        _validatable(whisper={"server-url": "http://127.0.0.1:8080"})
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"agent": {"backend": "codex", "draft": {"max-attempts": 0}}}, "max-attempts"),
        (
            {"agent": {"backend": "codex", "draft": {"preceding-context": -1}}},
            "preceding-context",
        ),
        (
            {"agent": {"backend": "codex", "draft": {"following-context": -1}}},
            "following-context",
        ),
        ({"agent": {"backend": "codex", "draft": {"workers": 0}}}, "workers"),
        (
            {"agent": {"backend": "codex", "draft": {"word-batch-size": 10}}},
            "word-batch-size",
        ),
        (
            {"agent": {"backend": "codex", "draft": {"word-following-context": -1}}},
            "word-following-context",
        ),
        (
            {"agent": {"backend": "codex", "draft": {"timeout-seconds": 0}}},
            "timeout-seconds",
        ),
        ({"agent": {"backend": "codex", "effort": "extreme"}}, "effort"),
        ({"subtitles": {"max-cue-seconds": 0}}, "max-cue-seconds"),
        ({"subtitles": {"max-cue-chars": 4}}, "max-cue-chars"),
        ({"elevenlabs": {"num-speakers": 0}}, "num-speakers"),
        ({"elevenlabs": {"num-speakers": 33}}, "num-speakers"),
        ({"whisper": {"stream-chunk-seconds": 0}}, "whisper.stream-chunk-seconds"),
        ({"whisper": {"stream-context-seconds": -1}}, "whisper.stream-context-seconds"),
        ({"output_mode": "both"}, "output-mode"),
    ],
)
def test_numeric_and_enum_settings_are_validated(
    overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_run_settings(_validatable(**overrides))


def test_review_open_command_is_required_for_open_modes() -> None:
    with pytest.raises(ValueError, match="review.open-command"):
        validate_run_settings(_validatable(review={"display-mode": "both"}))
    validate_run_settings(
        _validatable(
            review={"display-mode": "both", "open-command": "editor {file}"}
        )
    )


# --- ElevenLabs API key slot ---


def test_api_key_slot_is_exclusive_within_one_layer(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[elevenlabs]\napi-key = "sk"\napi-key-command = "pass show el"\n'
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        load_settings(user_file=MISSING, project_file=config)


def test_api_key_slot_replaces_as_a_unit_across_layers(tmp_path: Path) -> None:
    user = tmp_path / "user.toml"
    project = tmp_path / "project.toml"
    user.write_text('[elevenlabs]\napi-key-command = "pass show el"\n')
    project.write_text('[elevenlabs]\napi-key = "sk-project"\n')

    settings = load_settings(user_file=user, project_file=project)

    assert settings.elevenlabs.api_key == "sk-project"
    assert settings.elevenlabs.api_key_command is None
    assert settings.elevenlabs.api_key_from_cli is False


def test_c_override_is_the_highest_config_layer_for_the_slot(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project.toml"
    project.write_text('[elevenlabs]\napi-key-command = "false"\n')

    settings = load_settings(
        user_file=MISSING,
        project_file=project,
        cli_config={"elevenlabs": {"api-key": "sk-cli"}},
    )

    assert settings.elevenlabs.api_key == "sk-cli"
    assert settings.elevenlabs.api_key_command is None
    assert settings.elevenlabs.api_key_from_cli is True


def test_cli_slot_outranks_environment_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-env")
    settings = load_settings(
        user_file=MISSING,
        project_file=MISSING,
        cli_config={"elevenlabs": {"api-key": "sk-cli"}},
    )
    assert elevenlabs_api_key(settings) == "sk-cli"
    assert elevenlabs_api_key_source(settings) == "config"


def test_environment_variable_outranks_config_file_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-env")
    project = tmp_path / "project.toml"
    project.write_text('[elevenlabs]\napi-key = "sk-file"\n')
    settings = load_settings(user_file=MISSING, project_file=project)
    assert elevenlabs_api_key(settings) == "sk-env"
    assert elevenlabs_api_key_source(settings) == "env"


def test_config_file_slot_outranks_keyring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(elevenlabs_module, "_keyring_key", lambda: "sk-keyring")
    key_file = tmp_path / "key.txt"
    key_file.write_text("sk-from-file\n")
    settings = make_settings(
        elevenlabs={"api-key-file": str(key_file)}
    ).resolved()
    assert elevenlabs_api_key(settings) == "sk-from-file"
    assert elevenlabs_api_key_source(settings) == "file"


def test_keyring_is_the_last_resort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(elevenlabs_module, "_keyring_key", lambda: "sk-keyring")
    settings = make_settings().resolved()
    assert elevenlabs_api_key(settings) == "sk-keyring"
    assert elevenlabs_api_key_source(settings) == "keyring"


def test_keyring_key_is_fetched_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    class FakeKeyring:
        @staticmethod
        def get_password(service: str, entry: str) -> str:
            calls.append((service, entry))
            return "sk-cached"

    monkeypatch.setitem(sys.modules, "keyring", FakeKeyring())
    monkeypatch.setattr(elevenlabs_module, "_keyring_cached_key", None)
    assert elevenlabs_module._keyring_key() == "sk-cached"
    assert elevenlabs_module._keyring_key() == "sk-cached"
    assert len(calls) == 1


def test_keyring_key_missing_result_is_not_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only a found key is worth caching: a lookup that produced nothing may
    # be a transient Secret Service failure, and pinning it would turn every
    # later call into "no key" for the rest of the process.
    results = [None, "sk-late"]

    class FakeKeyring:
        @staticmethod
        def get_password(service: str, entry: str) -> str | None:
            return results.pop(0)

    monkeypatch.setitem(sys.modules, "keyring", FakeKeyring())
    monkeypatch.setattr(elevenlabs_module, "_keyring_cached_key", None)
    assert elevenlabs_module._keyring_key() is None
    assert elevenlabs_module._keyring_key() == "sk-late"


def test_missing_key_lists_every_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(elevenlabs_module, "_keyring_key", lambda: None)
    settings = make_settings().resolved()
    with pytest.raises(ValueError) as excinfo:
        elevenlabs_api_key(settings)
    message = str(excinfo.value)
    for source in (
        "-c elevenlabs.api-key",
        "ELEVENLABS_API_KEY",
        "api-key-file",
        "api-key-command",
        "yakiflow secret set elevenlabs",
    ):
        assert source in message
    assert elevenlabs_api_key_source(settings) == "not set"


def test_configured_key_file_that_fails_is_an_error_not_a_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(elevenlabs_module, "_keyring_key", lambda: "sk-keyring")
    settings = make_settings(
        elevenlabs={"api-key-file": str(tmp_path / "absent.txt")}
    ).resolved()
    with pytest.raises(ValueError, match="api-key-file"):
        elevenlabs_api_key(settings)


def test_configured_key_command_that_fails_is_an_error_not_a_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(elevenlabs_module, "_keyring_key", lambda: "sk-keyring")
    settings = make_settings(
        elevenlabs={"api-key-command": "exit 3"}
    ).resolved()
    with pytest.raises(ValueError, match="status 3"):
        elevenlabs_api_key(settings)


def test_key_command_output_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    settings = make_settings(
        elevenlabs={"api-key-command": "echo '  sk-cmd  '"}
    ).resolved()
    assert elevenlabs_api_key(settings) == "sk-cmd"


def test_settings_schema_version_is_three() -> None:
    assert SETTINGS_SCHEMA_VERSION == 3


def test_stream_mapping_and_agent_inheritance():
    settings = _validatable(agent={"backend": "codex", "draft": {"batch_size": 7},
                                   "stream": {"backend": "claude", "workers": 2}})
    assert settings.transcription.stream_backend == "whisper-server"
    assert settings.agent.stream.model == "sonnet"
    assert settings.agent.stream.batch_size == 7
    assert settings.agent.stream.workers == 2
    assert settings.for_stream().agent.draft.backend == "claude"
    assert _validatable(transcription={"backend": "elevenlabs"}).transcription.stream_backend == "elevenlabs-stream"
    assert settings.resolved() == settings


def test_old_stream_configuration_is_rejected(tmp_path):
    config = tmp_path / "config.toml"
    config.write_text("[stream]\nenabled = true\n")
    with pytest.raises(ValueError, match="single setting"):
        load_settings(user_file=MISSING, project_file=config)
    with pytest.raises(ValueError, match="realtime backend"):
        validate_run_settings(_validatable(transcription={"stream_backend": "elevenlabs"}))
