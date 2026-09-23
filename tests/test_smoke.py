import sys

import diffusion_nav

EXPECTED_PYTHON_VERSION = (3, 12)
EXPECTED_PACKAGE_VERSION = "0.1.0"


def test_project_uses_supported_python_version() -> None:
    """Verify that the test suite is running on the project's pinned Python release."""
    # Act
    actual_version = sys.version_info[:2]

    # Assert
    assert actual_version == EXPECTED_PYTHON_VERSION


def test_package_exposes_expected_version() -> None:
    """Verify that the package publishes its semantic version at the top level."""
    # Act
    actual_version = diffusion_nav.__version__

    # Assert
    assert actual_version == EXPECTED_PACKAGE_VERSION
