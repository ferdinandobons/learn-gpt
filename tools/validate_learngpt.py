"""
Changes compared with the previous files:
- This is not a lesson file and does not add model code.
- It validates the English public structure of `LearnGPT`.
- It requires one English course and rejects retired Italian guides.

File purpose:
- Check that `course_en.md`, `study/lessons`, `study/snapshots`,
  `final_project`, `tools`, and `data` stay consistent.
- Prevent a lesson from importing project code from another lesson snapshot.
- Ensure the course stays focused on explanations and snippets instead of full
  duplicated source files.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
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
    "data",
    "docs",
    "final_project",
    "study",
    "tests",
    "tools",
}

ALLOWED_LOCAL_ROOT_ENTRIES = {
    "checkpoints",
    "runs",
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
    "Batch creato",
    "Differenza rispetto",
    "Gruppi dell'optimizer",
    "Scopo del file",
    "Weight tying attivo",
    "Batch effettivo",
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

ITALIAN_PUBLIC_PATTERN = re.compile(
    r"\b(?:aggiunto|attenzione|caratteri|codice|contesto|corretto|corrente|"
    r"dati|dimensione|dopo|errore|esegui|esempio|esempi|futuro|indice|"
    r"lezione|migliore|modello|nessuno|numeri|numerici|perché|posizione|"
    r"prima|processato|quindi|risultato|salvato|seconda|stampa|stesso|"
    r"tensore|testo|trovato|tutti|ultima|valore|verificato|versione)\b|"
    r"[àèéìòù]",
    flags=re.IGNORECASE,
)

ITALIAN_COURSE_PATTERN = re.compile(
    r"\b(?:aggiornati|causale|codice|configurazione|contesto|corso|dati|"
    r"deve|devono|dopo|durante|esempio|esempi|lezione|lezioni|maschera|"
    r"migliore|modello|nuovo|ogni|opzionale|perché|poi|posizione|"
    r"preparazione|prima|punteggio|punteggi|quindi|salvato|somma|testo|"
    r"ultima|ultimo|valore|valori|vettore|vettori)\b",
    flags=re.IGNORECASE,
)

IGNORED_SCAN_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "venv",
}


def is_in_ignored_scan_dir(path: Path, project_dir: Path) -> bool:
    relative_parts = path.relative_to(project_dir).parts

    return any(part in IGNORED_SCAN_DIR_NAMES for part in relative_parts)


def project_python_files(project_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(project_dir.rglob("*.py"))
        if not is_in_ignored_scan_dir(path, project_dir)
    ]


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

    study_sample = project_dir / "data" / "study_sample.txt"
    if not study_sample.exists():
        errors.append(f"missing tracked study sample: {study_sample}")
    else:
        sample_text = read_text(study_sample)
        if len(sample_text) < 2_000:
            errors.append("data/study_sample.txt must contain at least 2,000 characters")
        if ITALIAN_PUBLIC_PATTERN.search(sample_text):
            errors.append("data/study_sample.txt must remain English-only")

    if not require_data:
        return

    processed_data_dir = project_dir / "data" / "processed" / "fineweb_edu"
    check_training_data_directory(processed_data_dir, errors)


def check_training_data_directory(data_dir: Path, errors: list[str]) -> None:
    if not data_dir.is_absolute():
        data_dir = data_dir.resolve()

    metadata_path = data_dir / "meta.json"
    train_path = data_dir / "train.bin"
    validation_path = data_dir / "val.bin"
    for path in (metadata_path, train_path, validation_path):
        if not path.exists():
            errors.append(f"missing prepared training data: {path}")
    if not all(path.exists() for path in (metadata_path, train_path, validation_path)):
        return

    try:
        metadata = json.loads(read_text(metadata_path))
    except (json.JSONDecodeError, OSError) as error:
        errors.append(f"invalid dataset metadata {metadata_path}: {error}")
        return

    if metadata.get("complete") is not True:
        errors.append(f"dataset metadata is not complete: {metadata_path}")
    if metadata.get("dtype") != "uint16":
        errors.append(f"dataset dtype must be uint16: {metadata_path}")
    if metadata.get("encoding_name") != "gpt2":
        errors.append(f"dataset encoding must be gpt2: {metadata_path}")

    counters = metadata.get("counters") or {}
    for split_name, path in (("train", train_path), ("val", validation_path)):
        token_count = counters.get(f"{split_name}_tokens")
        if not isinstance(token_count, int) or token_count < 1:
            errors.append(
                f"dataset metadata needs a positive {split_name}_tokens count: "
                f"{metadata_path}"
            )
            continue
        if path.stat().st_size != token_count * 2:
            errors.append(
                f"{path} size does not match {split_name}_tokens in meta.json"
            )


def check_markdown_basics(project_dir: Path, errors: list[str]) -> None:
    markdown_files = [
        "README.md",
        "course_en.md",
        "data/README.md",
        "docs/FINAL_TRAINING_RUNBOOK.md",
        "docs/VIDEO_SERIES_GUIDE.md",
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

        if ITALIAN_PUBLIC_PATTERN.search(text):
            errors.append(f"{name} must remain English-only")

    english_course = read_text(project_dir / "course_en.md")
    if "## Source Map" not in english_course:
        errors.append("course_en.md must contain the 'Source Map' section")
    if "## How to Run Study Scripts" not in english_course:
        errors.append("course_en.md must contain study script instructions")
    if "## How Study Snapshots Work" not in english_course:
        errors.append("course_en.md must explain study snapshots")
    if "Complete code" in english_course or "Complete study script code" in english_course:
        errors.append("course_en.md must not contain complete code sections")
    if "PDF" in english_course or "pdf" in english_course:
        errors.append("course_en.md must not reference PDF generation")

    italian_terms = sorted(
        {
            match.group(0).lower()
            for match in ITALIAN_COURSE_PATTERN.finditer(english_course)
        }
    )
    if italian_terms:
        errors.append(
            "course_en.md contains Italian terms: "
            f"{italian_terms}"
        )


def check_course_index(project_dir: Path, errors: list[str]) -> None:
    english_course = read_text(project_dir / "course_en.md")

    for lesson_number in lesson_numbers_from_study(project_dir):
        lesson = f"{lesson_number:02d}"
        if f"Lesson {lesson} -" not in english_course:
            errors.append(f"course_en.md does not contain index item Lesson {lesson}")
        if not re.search(rf"^## Lesson {lesson} - ", english_course, flags=re.MULTILINE):
            errors.append(f"course_en.md does not contain section ## Lesson {lesson}")


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

        if (
            lesson <= "35"
            and "DATASET_PATH =" in text
            and '"study_sample.txt"' not in text
        ):
            errors.append(f"{script}: must use the tracked data/study_sample.txt")

        if "/private/tmp" in text:
            errors.append(f"{script}: contains a non-portable /private/tmp path")

    lesson_20_model = read_text(snapshots_dir(project_dir) / "lesson_20" / "model.py")
    lesson_21_model = read_text(snapshots_dir(project_dir) / "lesson_21" / "model.py")
    if "output_projection" in lesson_20_model:
        errors.append("lesson_20 must stop at multi-head concatenation")
    if "output_projection" not in lesson_21_model:
        errors.append("lesson_21 must introduce the attention output projection")

    lesson_36_scripts = list(lessons_dir(project_dir).glob("36_*.py"))
    if lesson_36_scripts:
        lesson_36_text = read_text(lesson_36_scripts[0])
        for required_term in (
            "get_vocabulary_size",
            "load_training_and_validation_data",
            "Optimizer groups",
        ):
            if required_term not in lesson_36_text:
                errors.append(f"lesson 36 must demonstrate {required_term}")

    lesson_42_scripts = list(lessons_dir(project_dir).glob("42_*.py"))
    if lesson_42_scripts:
        lesson_42_text = read_text(lesson_42_scripts[0])
        if "data/processed" in lesson_42_text or '"processed"' in lesson_42_text:
            errors.append("lesson 42 smoke test must not require processed training data")


def check_english_code_identifiers(project_dir: Path, errors: list[str]) -> None:
    for path in project_python_files(project_dir):
        text = read_text(path)

        try:
            module = ast.parse(text)
        except SyntaxError:
            continue

        old_names = sorted(collect_identifiers(module) & FORBIDDEN_ITALIAN_IDENTIFIERS)
        if old_names:
            errors.append(f"{path}: non-English Python identifiers: {old_names}")


def check_python_text_is_english(project_dir: Path, errors: list[str]) -> None:
    for path in project_python_files(project_dir):
        if path == project_dir / "tools" / "validate_learngpt.py":
            continue

        text = read_text(path)
        found = sorted(marker for marker in ITALIAN_CODE_MARKERS if marker in text)
        if found:
            errors.append(f"{path}: Italian public text remains: {found}")
        if ITALIAN_PUBLIC_PATTERN.search(text):
            errors.append(f"{path}: Italian public text remains")


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
    course = read_text(project_dir / "course_en.md")
    headings = list(re.finditer(r"^## (.+)$", course, flags=re.MULTILINE))

    for index, heading in enumerate(headings):
        title = heading.group(1)
        lesson_match = re.match(r"Lesson (\d{2}) - ", title)
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


def check_final_project_snapshot(project_dir: Path, errors: list[str]) -> None:
    final_dir = project_dir / "final_project"
    snapshot_dir = snapshots_dir(project_dir) / "lesson_42"

    if not final_dir.is_dir() or not snapshot_dir.is_dir():
        return

    final_files = {
        path.name
        for path in final_dir.iterdir()
        if path.suffix == ".py" or path.name.startswith("requirements")
    }
    snapshot_files = {
        path.name
        for path in snapshot_dir.iterdir()
        if path.suffix == ".py" or path.name.startswith("requirements")
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
    pycache_dirs = [
        path
        for path in sorted(project_dir.rglob("__pycache__"))
        if not is_in_ignored_scan_dir(path, project_dir)
    ]
    for path in pycache_dirs:
        errors.append(f"__pycache__ directory present: {path}")


def check_operational_guides(project_dir: Path, errors: list[str]) -> None:
    runbook_path = project_dir / "docs" / "FINAL_TRAINING_RUNBOOK.md"
    video_guide_path = project_dir / "docs" / "VIDEO_SERIES_GUIDE.md"
    workflow_path = project_dir / "docs" / "training_workflow.json"

    for path in (runbook_path, video_guide_path, workflow_path):
        if not path.exists():
            errors.append(f"missing operational guide: {path}")
    if not all(path.exists() for path in (runbook_path, video_guide_path, workflow_path)):
        return

    runbook = read_text(runbook_path)
    required_runbook_terms = (
        "Windows PowerShell",
        "Apple Silicon MPS",
        "--training-steps 45000",
        "--context-size 256",
        "--num-transformer-blocks 6",
        "--training-data-dir",
        "--resume-checkpoint-path",
        "--seed 1337",
        "dataset fingerprint",
        "amp_overflows",
        "base language model",
    )
    for term in required_runbook_terms:
        if term not in runbook:
            errors.append(f"final training runbook is missing: {term}")
    for stale_term in (
        "cu128",
        "learngpt-cuda.pt",
        "--context-size 128",
        "--eval-interval 20",
    ):
        if stale_term in runbook:
            errors.append(f"final training runbook contains stale profile: {stale_term}")

    video_guide = read_text(video_guide_path)
    for episode in range(1, 11):
        if f"## Episode {episode} " not in video_guide:
            errors.append(f"video series guide is missing Episode {episode}")

    try:
        workflow = json.loads(read_text(workflow_path))
    except json.JSONDecodeError as error:
        errors.append(f"invalid training workflow JSON: {error}")
        return

    model = workflow.get("model") or {}
    expected_model = {
        "parameters": 17_716_049,
        "contextSize": 256,
        "embeddingSize": 256,
        "heads": 4,
        "blocks": 6,
        "vocabularySize": 50_257,
        "effectiveTokensPerStep": 8_192,
        "trainingSteps": 45_000,
    }
    for name, expected_value in expected_model.items():
        if model.get(name) != expected_value:
            errors.append(
                f"training workflow model.{name} must be {expected_value}"
            )

    steps = workflow.get("steps") or []
    if [step.get("number") for step in steps] != list(range(1, 11)):
        errors.append("training workflow must contain consecutive steps 1 through 10")
    if not workflow.get("healthSignals") or not workflow.get("stopSignals"):
        errors.append("training workflow must define healthSignals and stopSignals")
    for resource in workflow.get("resources") or []:
        resource_path = project_dir / resource.get("path", "")
        if not resource_path.is_file():
            errors.append(f"training workflow resource does not exist: {resource_path}")

    verified_run_path = (
        project_dir / "docs" / "verified_runs" / "mps-18m-1g-45000.json"
    )
    try:
        verified_run = json.loads(read_text(verified_run_path))
    except (json.JSONDecodeError, OSError) as error:
        errors.append(f"invalid verified run manifest: {error}")
        return

    source_revision = verified_run.get("sourceRevisionAtTraining", "")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        errors.append("verified run sourceRevisionAtTraining must be a full Git SHA")
        return

    try:
        shallow_result = subprocess.run(
            ["git", "rev-parse", "--is-shallow-repository"],
            cwd=project_dir,
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return

    if shallow_result.returncode == 0 and shallow_result.stdout.strip() == "false":
        commit_result = subprocess.run(
            ["git", "cat-file", "-e", f"{source_revision}^{{commit}}"],
            cwd=project_dir,
            capture_output=True,
            check=False,
        )
        if commit_result.returncode != 0:
            errors.append(
                "verified run sourceRevisionAtTraining does not resolve to a commit"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate LearnGPT structure, lessons, snapshots, and docs.",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Also check local untracked datasets.",
    )
    parser.add_argument(
        "--training-data-dir",
        type=Path,
        action="append",
        default=[],
        help="Validate one prepared token directory; may be supplied more than once.",
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
    check_final_project_snapshot(project_dir, errors)
    check_english_code_identifiers(project_dir, errors)
    check_python_text_is_english(project_dir, errors)
    check_operational_guides(project_dir, errors)
    for data_dir in args.training_data_dir:
        resolved_data_dir = data_dir if data_dir.is_absolute() else project_dir / data_dir
        check_training_data_directory(resolved_data_dir, errors)
    check_no_pycache(project_dir, errors)

    if errors:
        fail(errors)

    print("OK: structure, courses, snapshots, and scripts are consistent.")


if __name__ == "__main__":
    main()
