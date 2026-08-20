from __future__ import annotations

import sys
from unittest.mock import patch

from reach_touch_inspire_demo import parse_args


def parse_with(arguments: list[str]):
    with patch.object(sys, "argv", ["reach_touch_inspire_demo.py", *arguments]):
        return parse_args()


def test_keep_viewer_open_defaults_to_false() -> None:
    default_args = parse_with(["--no-viewer"])
    assert not default_args.keep_viewer_open


def test_keep_viewer_open_can_be_enabled() -> None:
    keep_open_args = parse_with(["--no-viewer", "--keep-viewer-open"])
    assert keep_open_args.keep_viewer_open


def main() -> None:
    test_keep_viewer_open_defaults_to_false()
    test_keep_viewer_open_can_be_enabled()
    print("Reach-touch demo argument tests passed.")


if __name__ == "__main__":
    main()
