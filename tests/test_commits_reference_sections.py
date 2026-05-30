import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_log_subjects() -> list[str]:
    out = subprocess.check_output(
        ["git", "log", "--pretty=format:%s"],
        cwd=REPO_ROOT,
        text=True,
    )
    return [line for line in out.splitlines() if line.strip()]


def test_at_least_one_commit_present() -> None:
    """Sanity: git history exists."""
    subjects = _git_log_subjects()
    assert len(subjects) >= 1, "git log returned 0 commits; repo not initialised?"


def test_every_commit_starts_with_phase_n() -> None:
    """CLAUDE.md commit-message convention: '<phase>: <§-ref> <imperative summary>'.

    For Phase 0 and onward, the subject must start with 'Phase N' (or 'Phase N fix').
    This binds every committed change to a § of the build plan — the §1.4
    architect/implementer evidence trail.
    """
    subjects = _git_log_subjects()
    bad = [s for s in subjects if not re.match(r"^Phase \d+( fix)?\b", s)]
    assert not bad, "These commit subjects don't start with 'Phase N': " + "\n".join(f"  - {s}" for s in bad)
