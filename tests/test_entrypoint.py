"""The project switches in main.py retain their original skip-run behavior."""

from utils import skip_run


def test_project_blocks_can_be_run_or_skipped():
    visited = []
    with skip_run("run", "active") as check, check():
        visited.append("active")
    with skip_run("skip", "inactive") as check, check():
        visited.append("inactive")
    assert visited == ["active"]
