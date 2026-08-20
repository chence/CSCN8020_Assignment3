from __future__ import annotations

import sys
from unittest.mock import patch

from evaluate_actor_critic import parse_args


def parse_with(arguments: list[str]):
    with patch.object(sys, "argv", ["evaluate_actor_critic.py", *arguments]):
        return parse_args()


def test_keep_viewer_open_defaults_to_false() -> None:
    default_args = parse_with([])
    assert not default_args.keep_viewer_open


def test_keep_viewer_open_can_be_enabled_with_viewer() -> None:
    keep_open_args = parse_with(["--viewer", "--keep-viewer-open"])
    assert keep_open_args.viewer
    assert keep_open_args.keep_viewer_open


def main() -> None:
    test_keep_viewer_open_defaults_to_false()
    test_keep_viewer_open_can_be_enabled_with_viewer()
    print("Actor-Critic evaluation argument tests passed.")


if __name__ == "__main__":
    main()
