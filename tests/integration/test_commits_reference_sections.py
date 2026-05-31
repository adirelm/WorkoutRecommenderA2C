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


_ALLOWED_PREFIX_RE = re.compile(
    r"^("
    r"Phase \d+( fix(-execution)?| gate-fix(-\d+)?| completion| ruff-fix)?"  # phase work (the §1.4 trail)
    r"|PII scrub"  # emergency PII redactions
    r"|Revert "  # git revert auto-generated subjects
    r"|Merge "  # git merge auto-generated subjects
    r")\b"
)


def test_every_commit_starts_with_phase_n() -> None:
    """CLAUDE.md commit-message convention: '<phase>: <§-ref> <imperative summary>'.

    Every commit subject must start with one of the allowed prefixes:
      * ``Phase N`` / ``Phase N fix`` / ``Phase N gate-fix`` / ``Phase N completion`` —
        phase-numbered work (the §1.4 architect/implementer evidence trail).
      * ``PII scrub`` — emergency redactions that intentionally sit outside the
        phase sequence (cover sheet, accidental real-name leaks, etc.).
      * ``Revert ...`` / ``Merge ...`` — git-auto-generated subjects we cannot rewrite
        without force-pushing main (forbidden per project rules).
    """
    subjects = _git_log_subjects()
    bad = [s for s in subjects if not _ALLOWED_PREFIX_RE.match(s)]
    assert not bad, "These commit subjects don't match an allowed prefix: " + "\n".join(
        f"  - {s}" for s in bad
    )
