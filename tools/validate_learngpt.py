"""
Changes compared with the previous files:
- This is not a lesson file and does not add model code.
- It validates the English public structure of `LearnGPT`.
- It no longer requires `guidance.md`, which is intentionally local-only.

File purpose:
- Check that `course_it.md`, `course_en.md`, `study/lessons`,
  `study/snapshots`, `final_project`, `tools`, and `data` stay consistent.
- Prevent a lesson from importing project code from another lesson snapshot.
- Check that complete code blocks in the course stay aligned with real files.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path


OLD_MODEL_NAMES = {
    "BigramLanguageModel",
    "TokenEmbeddingLanguageModel",
    "PositionEmbeddingLanguageModel",
    "SingleHeadAttentionLanguageModel",
    "MultiHeadAttentionLanguageModel",
    "ResidualAttentionLanguageModel",
    "LayerNormAttentionLanguageModel",
    "FeedForwardLanguageModel",
    "TransformerBlockLanguageModel",
    "MultiBlockTransformerLanguageModel",
    "FinalLayerNormTransformerLanguageModel",
}

EXPECTED_ROOT_ENTRIES = {
    "README.md",
    "course_en.md",
    "course_it.md",
    "data",
    "final_project",
    "study",
    "tools",
}

ALLOWED_LOCAL_ROOT_ENTRIES = {
    "guidance.md",
}

FORBIDDEN_ITALIAN_IDENTIFIERS = {
    "allena_modello",
    "carattere",
    "carattere_a_id",
    "caratteri_unici",
    "contesto",
    "contesto_testo",
    "controllo_input",
    "controllo_target",
    "crea_batch",
    "crea_esempio",
    "crea_vocabolario",
    "differenza_parametro",
    "esempio",
    "id_a_carattere",
    "input_testo",
    "modello",
    "mostra_previsione",
    "numero",
    "numero_esempio",
    "numeri",
    "posizione",
    "posizione_iniziale",
    "primo_predicted_token",
    "primo_embedding_sommato",
    "primo_input",
    "primo_parametro_dopo",
    "primo_parametro_prima",
    "primo_target",
    "primo_token",
    "primo_token_embedding",
    "primo_token_id",
    "primo_token_previsto",
    "prossimo_carattere",
    "prossimo_token",
    "punto_di_taglio",
    "punteggi_first_token",
    "punteggi_primo_token",
    "target_testo",
    "tensore",
    "testo",
    "testo_completo",
    "testo_generato",
    "testo_ricostruito",
    "token_previsto",
}

FORBIDDEN_PUBLIC_REFERENCES = {
    "corso.md",
    "guidance.md",
    "progetto_finale",
    "strumenti",
    "studio/lezioni",
    "studio/snapshot",
    "studio.snapshot",
    "lezione_",
}

ITALIAN_CODE_MARKERS = {
    "Differenza rispetto",
    "Scopo del file",
    "Prima ",
    "Qui ",
    "Questa ",
    "Questo ",
    "lezione",
    "percorso",
    "manca",
    "deve",
    "non può",
    "non è",
}


def lesson_numbers_from_study(project_dir: Path) -> list[int]:
    numbers: list[int] = []

    for script in sorted(lessons_dir(project_dir).glob("[0-9][0-9]_*.py")):
        numbers.append(int(script.name[:2]))

    return numbers


def lessons_dir(project_dir: Path) -> Path:
    return project_dir / "study" / "lessons"


def snapshots_dir(project_dir: Path) -> Path:
    return project_dir / "study" / "snapshots"


def fail(errors: list[str]) -> None:
    for error in errors:
        print(f"ERROR: {error}")
    raise SystemExit(1)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def collect_identifiers(module: ast.Module) -> set[str]:
    identifiers: set[str] = set()

    for node in ast.walk(module):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            identifiers.add(node.name)
        elif isinstance(node, ast.arg):
            identifiers.add(node.arg)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)

    return identifiers


def check_required_structure(
    project_dir: Path,
    errors: list[str],
    require_data: bool,
) -> None:
    for name in sorted(EXPECTED_ROOT_ENTRIES):
        if not (project_dir / name).exists():
            errors.append(f"missing {project_dir / name}")

    if not lessons_dir(project_dir).is_dir():
        errors.append(f"missing {lessons_dir(project_dir)}")

    if not snapshots_dir(project_dir).is_dir():
        errors.append(f"missing {snapshots_dir(project_dir)}")

    allowed_root_entries = EXPECTED_ROOT_ENTRIES | ALLOWED_LOCAL_ROOT_ENTRIES
    for entry in sorted(project_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.name not in allowed_root_entries:
            errors.append(f"unexpected item in LearnGPT root: {entry.name}")

    lesson_numbers = lesson_numbers_from_study(project_dir)

    if not lesson_numbers:
        errors.append("at least one numbered script is required in study/lessons")
        return

    expected_numbers = list(range(1, max(lesson_numbers) + 1))
    if lesson_numbers != expected_numbers:
        errors.append(
            "lesson scripts must be consecutive from 01 to "
            f"{max(lesson_numbers):02d}"
        )

    for lesson_number in expected_numbers:
        lesson = f"{lesson_number:02d}"
        script_matches = list(lessons_dir(project_dir).glob(f"{lesson}_*.py"))
        if len(script_matches) != 1:
            errors.append(f"lesson {lesson} must have one script in study/lessons")

        snapshot_dir = snapshots_dir(project_dir) / f"lesson_{lesson}"
        if not snapshot_dir.is_dir():
            errors.append(f"missing snapshot {snapshot_dir}")

    data_readme = project_dir / "data" / "README.md"
    if not data_readme.exists():
        errors.append(f"missing data guide: {data_readme}")

    if not require_data:
        return

    sample_path = project_dir / "data" / "raw" / "fineweb_edu_sample.txt"
    if not sample_path.exists():
        errors.append(f"missing shared text sample: {sample_path}")

    processed_data_dir = project_dir / "data" / "processed" / "fineweb_edu"
    for filename in ["train.bin", "val.bin", "meta.json"]:
        data_path = processed_data_dir / filename
        if not data_path.exists():
            errors.append(f"missing processed FineWeb-Edu dataset: {data_path}")


def check_markdown_basics(project_dir: Path, errors: list[str]) -> None:
    markdown_files = [
        "README.md",
        "course_it.md",
        "course_en.md",
        "data/README.md",
        "study/snapshots/README.md",
    ]

    for name in markdown_files:
        path = project_dir / name
        if not path.exists():
            continue
        text = read_text(path)
        if not text.startswith("# "):
            errors.append(f"{name} must start with an H1 title")
        if text.count("```") % 2 != 0:
            errors.append(f"{name} contains unbalanced code fences")
        if re.search(r"\b(TODO|FIXME|TBD)\b", text, re.IGNORECASE):
            errors.append(f"{name} contains TODO/FIXME/TBD")

        legacy_title = "D" + "ivina" + " " + "C" + "ommedia"
        legacy_filename = "div" + "ina" + "_" + "com" + "media"
        if legacy_title in text or legacy_filename in text:
            errors.append(f"{name} still references the old dataset")

        for reference in FORBIDDEN_PUBLIC_REFERENCES:
            if reference in text:
                errors.append(f"{name} still contains legacy reference: {reference}")

    italian_course = read_text(project_dir / "course_it.md")
    if "## Mappa delle fonti" not in italian_course:
        errors.append("course_it.md must contain the 'Mappa delle fonti' section")
    if "## Navigazione del documento" not in italian_course:
        errors.append("course_it.md must contain the 'Navigazione del documento' section")

    english_course = read_text(project_dir / "course_en.md")
    if "## Source Map" not in english_course:
        errors.append("course_en.md must contain the 'Source Map' section")
    if "## Document Navigation" not in english_course:
        errors.append("course_en.md must contain the 'Document Navigation' section")


def check_course_index(project_dir: Path, errors: list[str]) -> None:
    italian_course = read_text(project_dir / "course_it.md")

    for lesson_number in lesson_numbers_from_study(project_dir):
        lesson = f"{lesson_number:02d}"
        if f"Lezione {lesson} -" not in italian_course:
            errors.append(f"course_it.md does not contain index item Lezione {lesson}")
        if not re.search(rf"^## Lezione {lesson} - ", italian_course, flags=re.MULTILINE):
            errors.append(f"course_it.md does not contain section ## Lezione {lesson}")

    english_course = read_text(project_dir / "course_en.md")
    if "Lesson 42 - Final Project" not in english_course:
        errors.append("course_en.md must contain Lesson 42 - Final Project")


def check_study_scripts(project_dir: Path, errors: list[str]) -> None:
    for script in sorted(lessons_dir(project_dir).glob("[0-9][0-9]_*.py")):
        lesson = script.name[:2]
        text = read_text(script)

        try:
            module = ast.parse(text)
        except SyntaxError as exc:
            errors.append(f"{script}: invalid syntax: {exc}")
            continue

        docstring = ast.get_docstring(module) or ""
        if "Changes compared" not in docstring:
            errors.append(f"{script}: missing 'Changes compared' in module docstring")
        if "File purpose" not in docstring:
            errors.append(f"{script}: missing 'File purpose' in module docstring")

        if lesson >= "12":
            expected = f"from study.snapshots.lesson_{lesson}.model import LanguageModel"
            if expected not in text:
                errors.append(f"{script}: model import is not aligned with lesson_{lesson}")

        dataset_path_lines = [
            line for line in text.splitlines() if line.startswith("DATASET_PATH =")
        ]
        for line in dataset_path_lines:
            if '"snapshots"' in line and f'"lesson_{lesson}"' not in line:
                errors.append(f"{script}: DATASET_PATH does not point to lesson_{lesson}")


def check_english_code_identifiers(project_dir: Path, errors: list[str]) -> None:
    for path in sorted(project_dir.glob("**/*.py")):
        text = read_text(path)

        try:
            module = ast.parse(text)
        except SyntaxError:
            continue

        old_names = sorted(collect_identifiers(module) & FORBIDDEN_ITALIAN_IDENTIFIERS)
        if old_names:
            errors.append(f"{path}: non-English Python identifiers: {old_names}")


def check_python_text_is_english(project_dir: Path, errors: list[str]) -> None:
    for path in sorted(project_dir.glob("**/*.py")):
        if path == project_dir / "tools" / "validate_learngpt.py":
            continue

        text = read_text(path)
        found = sorted(marker for marker in ITALIAN_CODE_MARKERS if marker in text)
        if found:
            errors.append(f"{path}: Italian public text remains: {found}")


def check_models(project_dir: Path, errors: list[str]) -> None:
    model_paths = sorted(snapshots_dir(project_dir).glob("lesson_*/model.py"))
    model_paths.append(project_dir / "final_project" / "model.py")

    for path in model_paths:
        text = read_text(path)

        try:
            module = ast.parse(text)
        except SyntaxError as exc:
            errors.append(f"{path}: invalid syntax: {exc}")
            continue

        classes = [node.name for node in module.body if isinstance(node, ast.ClassDef)]
        if classes.count("LanguageModel") != 1:
            errors.append(f"{path}: must contain exactly one LanguageModel class")

        old_names = sorted(set(classes) & OLD_MODEL_NAMES)
        if old_names:
            errors.append(f"{path}: contains old model class names: {old_names}")

        docstring = ast.get_docstring(module) or ""
        if "Changes compared" not in docstring:
            errors.append(f"{path}: missing 'Changes compared' in module docstring")
        if "File purpose" not in docstring:
            errors.append(f"{path}: missing 'File purpose' in module docstring")


def check_course_lesson_references(project_dir: Path, errors: list[str]) -> None:
    course = read_text(project_dir / "course_it.md")
    headings = list(re.finditer(r"^## (.+)$", course, flags=re.MULTILINE))

    for index, heading in enumerate(headings):
        title = heading.group(1)
        lesson_match = re.match(r"Lezione (\d{2}) - ", title)
        if not lesson_match:
            continue

        lesson = lesson_match.group(1)
        end = headings[index + 1].start() if index + 1 < len(headings) else len(course)
        body = course[heading.start() : end]
        refs = re.findall(
            r"study/snapshots[\" /]+lesson_(\d{2})|study\.snapshots\.lesson_(\d{2})",
            body,
        )
        found = {left or right for left, right in refs}
        wrong = sorted(number for number in found if number != lesson)
        if wrong:
            errors.append(f"Lesson {lesson}: references different snapshots: {wrong}")


def check_complete_code_blocks(project_dir: Path, errors: list[str]) -> None:
    course = read_text(project_dir / "course_it.md")
    patterns = [
        re.compile(
            r"### Codice completo(?: aggiornato)?: `(?P<path>[^`]+)`\n\n"
            r"```python\n(?P<code>.*?)\n```",
            re.DOTALL,
        ),
        re.compile(
            r"File:\n\n```text\n(?P<path>LearnGPT/[^`\n]+?\.py)\n```\n\n"
            r"Codice:\n\n```python\n(?P<code>.*?)\n```",
            re.DOTALL,
        ),
    ]

    for pattern in patterns:
        for match in pattern.finditer(course):
            raw_path = match.group("path").strip()
            code = match.group("code").strip() + "\n"

            if raw_path.startswith("LearnGPT/"):
                file_path = project_dir.parent / raw_path
            elif raw_path.startswith(("study/", "final_project/", "tools/")):
                file_path = project_dir / raw_path
            else:
                continue

            if not file_path.exists():
                errors.append(f"course_it.md references missing file: {raw_path}")
                continue

            actual = read_text(file_path).strip() + "\n"
            if code != actual:
                errors.append(f"course_it.md is not aligned with {raw_path}")


def check_final_project_snapshot(project_dir: Path, errors: list[str]) -> None:
    final_dir = project_dir / "final_project"
    snapshot_dir = snapshots_dir(project_dir) / "lesson_42"

    if not final_dir.is_dir() or not snapshot_dir.is_dir():
        return

    final_files = {
        path.name
        for path in final_dir.iterdir()
        if path.suffix == ".py" or path.name == "requirements.txt"
    }
    snapshot_files = {
        path.name
        for path in snapshot_dir.iterdir()
        if path.suffix == ".py" or path.name == "requirements.txt"
    }

    if final_files != snapshot_files:
        missing = sorted(final_files - snapshot_files)
        extra = sorted(snapshot_files - final_files)
        if missing:
            errors.append(f"lesson_42 does not contain final files: {missing}")
        if extra:
            errors.append(f"lesson_42 contains files not present in final project: {extra}")
        return

    for filename in sorted(final_files):
        final_text = read_text(final_dir / filename)
        snapshot_text = read_text(snapshot_dir / filename)
        if final_text != snapshot_text:
            errors.append(
                "lesson_42 is not aligned with final_project for "
                f"{filename}"
            )


def check_no_pycache(project_dir: Path, errors: list[str]) -> None:
    pycache_dirs = sorted(project_dir.rglob("__pycache__"))
    for path in pycache_dirs:
        errors.append(f"__pycache__ directory present: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate LearnGPT structure, lessons, snapshots, and docs.",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Also check local untracked datasets.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_dir = Path(__file__).resolve().parent.parent
    errors: list[str] = []

    check_required_structure(project_dir, errors, require_data=args.require_data)
    check_markdown_basics(project_dir, errors)
    check_course_index(project_dir, errors)
    check_study_scripts(project_dir, errors)
    check_models(project_dir, errors)
    check_course_lesson_references(project_dir, errors)
    check_complete_code_blocks(project_dir, errors)
    check_final_project_snapshot(project_dir, errors)
    check_english_code_identifiers(project_dir, errors)
    check_python_text_is_english(project_dir, errors)
    check_no_pycache(project_dir, errors)

    if errors:
        fail(errors)

    print("OK: structure, courses, snapshots, and scripts are consistent.")


if __name__ == "__main__":
    main()
