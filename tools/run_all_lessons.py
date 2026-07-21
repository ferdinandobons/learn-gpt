"""Run every LearnGPT lesson with the current Python interpreter.

This clean-clone gate deliberately does not use local processed training data.
It proves that the tracked study sample and the self-contained final smoke test
are enough to execute the complete teaching path.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    lessons = sorted((project_dir / "study" / "lessons").glob("[0-9][0-9]_*.py"))

    if len(lessons) != 42:
        raise RuntimeError(f"Expected 42 lesson scripts, found {len(lessons)}.")

    for lesson in lessons:
        relative_path = lesson.relative_to(project_dir)
        print(f"RUN {relative_path}", flush=True)
        subprocess.run(
            [sys.executable, "-B", str(lesson)],
            cwd=project_dir,
            check=True,
        )

    print("OK: all 42 lessons passed.")


if __name__ == "__main__":
    main()
