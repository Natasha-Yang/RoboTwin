import re

import pytest

from openpi.training import episode_selection


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


def _stub_matchers(monkeypatch: pytest.MonkeyPatch, matchers: list[tuple[str, re.Pattern]]) -> None:
    monkeypatch.setattr(episode_selection, "_task_matchers", lambda: matchers)


def test_task_hint_resolves_single_task_dataset_with_ambiguous_templates(monkeypatch: pytest.MonkeyPatch):
    peg_instruction = r"^(?:Put|Drop) peg .*socket.*\.?$"
    matchers = [
        ("insert_peg_socket_loose", _rx(peg_instruction)),
        ("insert_peg_socket_med", _rx(peg_instruction)),
        ("insert_peg_socket_tight", _rx(peg_instruction)),
        ("place_bread_basket", _rx(r"^Drop .*$")),
    ]
    _stub_matchers(monkeypatch, matchers)

    got = episode_selection.assign_episode_tasks(
        [
            "Put peg in the square socket.",
            "Drop peg into the gray socket.",
        ],
        task_hint="insert_peg_socket_loose",
    )

    assert got == ["insert_peg_socket_loose", "insert_peg_socket_loose"]


def test_ambiguous_templates_without_hint_still_raise(monkeypatch: pytest.MonkeyPatch):
    matchers = [
        ("insert_peg_socket_loose", _rx(r"^Put peg .*$")),
        ("insert_peg_socket_med", _rx(r"^Put peg .*$")),
    ]
    _stub_matchers(monkeypatch, matchers)

    with pytest.raises(ValueError, match="ambiguous"):
        episode_selection.assign_episode_tasks(["Put peg in the square socket."])


def test_episode_task_labels_are_unique_anchors(monkeypatch: pytest.MonkeyPatch):
    matchers = [
        ("insert_peg_socket_loose", _rx(r"^does not match$")),
        ("insert_peg_socket_med", _rx(r"^does not match$")),
    ]
    _stub_matchers(monkeypatch, matchers)

    got = episode_selection.assign_episode_tasks(["insert_peg_socket_loose", "insert peg socket loose."])

    assert got == ["insert_peg_socket_loose", "insert_peg_socket_loose"]
