import sys

import diffusion_nav


def test_python_version() -> None:
    assert sys.version_info[:2] == (3, 12)


def test_package_version() -> None:
    assert diffusion_nav.__version__ == "0.1.0"
