"""
Differenza rispetto ai file precedenti:
- Questo non è un file di lezione e non aggiunge codice al modello.
- Aggiunge un controllo automatico pensato per la struttura specifica di
  `LearnGPT`.

Scopo del file:
- Verificare che `corso.md`, `guidance.md`, `studio/lezioni`,
  `studio/snapshot`, `progetto_finale` e `data` restino coerenti tra loro.
- Evitare che una lezione importi per errore codice di un'altra lezione.
- Controllare che i blocchi di codice completi nel corso restino allineati ai
  file reali.
"""

from __future__ import annotations

import ast
import argparse
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
    "corso.md",
    "data",
    "guidance.md",
    "progetto_finale",
    "strumenti",
    "studio",
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


def lesson_numbers_from_studio(project_dir: Path) -> list[int]:
    numbers: list[int] = []

    for script in sorted(lessons_dir(project_dir).glob("[0-9][0-9]_*.py")):
        numbers.append(int(script.name[:2]))

    return numbers


def lessons_dir(project_dir: Path) -> Path:
    return project_dir / "studio" / "lezioni"


def snapshots_dir(project_dir: Path) -> Path:
    return project_dir / "studio" / "snapshot"


def fail(errors: list[str]) -> None:
    for error in errors:
        print(f"ERRORE: {error}")
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
            errors.append(f"manca {project_dir / name}")

    if not lessons_dir(project_dir).is_dir():
        errors.append(f"manca {lessons_dir(project_dir)}")

    if not snapshots_dir(project_dir).is_dir():
        errors.append(f"manca {snapshots_dir(project_dir)}")

    for entry in sorted(project_dir.iterdir()):
        if entry.name.startswith("."):
            continue
        if entry.name not in EXPECTED_ROOT_ENTRIES:
            errors.append(f"elemento inatteso nella root LearnGPT: {entry.name}")

    lesson_numbers = lesson_numbers_from_studio(project_dir)

    if not lesson_numbers:
        errors.append("manca almeno uno script numerato in studio/lezioni")
        return

    expected_numbers = list(range(1, max(lesson_numbers) + 1))
    if lesson_numbers != expected_numbers:
        errors.append(
            "gli script in studio devono essere consecutivi da 01 a "
            f"{max(lesson_numbers):02d}"
        )

    for lesson_number in expected_numbers:
        lesson = f"{lesson_number:02d}"
        script_matches = list(lessons_dir(project_dir).glob(f"{lesson}_*.py"))
        if len(script_matches) != 1:
            errors.append(f"la lezione {lesson} deve avere uno script in studio/lezioni")

        snapshot_dir = snapshots_dir(project_dir) / f"lezione_{lesson}"
        if not snapshot_dir.is_dir():
            errors.append(f"manca lo snapshot {snapshot_dir}")
            continue

    data_readme = project_dir / "data" / "README.md"
    if not data_readme.exists():
        errors.append(f"manca la guida ai dati: {data_readme}")

    if not require_data:
        return

    sample_path = project_dir / "data" / "raw" / "fineweb_edu_sample.txt"
    if not sample_path.exists():
        errors.append(f"manca il campione testuale condiviso: {sample_path}")

    processed_data_dir = project_dir / "data" / "processed" / "fineweb_edu"
    for filename in ["train.bin", "val.bin", "meta.json"]:
        data_path = processed_data_dir / filename
        if not data_path.exists():
            errors.append(f"manca il dataset FineWeb-Edu processato: {data_path}")


def check_markdown_basics(project_dir: Path, errors: list[str]) -> None:
    for name in ["corso.md", "guidance.md"]:
        path = project_dir / name
        if not path.exists():
            continue
        text = read_text(path)
        if not text.startswith("# "):
            errors.append(f"{name} deve iniziare con un titolo H1")
        if text.count("```") % 2 != 0:
            errors.append(f"{name} contiene blocchi di codice non bilanciati")
        if re.search(r"\b(TODO|FIXME|TBD)\b", text, re.IGNORECASE):
            errors.append(f"{name} contiene TODO/FIXME/TBD")
        if "progetto_lezioni" in text:
            errors.append(f"{name} contiene ancora riferimenti a progetto_lezioni")
        legacy_title = "D" + "ivina" + " " + "C" + "ommedia"
        legacy_filename = "div" + "ina" + "_" + "com" + "media"
        if legacy_title in text or legacy_filename in text:
            errors.append(f"{name} contiene ancora riferimenti al vecchio dataset")

    course = read_text(project_dir / "corso.md")
    if "## Mappa delle fonti" not in course:
        errors.append("corso.md deve contenere la sezione 'Mappa delle fonti'")
    if "## Navigazione del documento" not in course:
        errors.append("corso.md deve contenere la sezione 'Navigazione del documento'")

    guidance = read_text(project_dir / "guidance.md")
    if "## Fonti e riferimenti" not in guidance:
        errors.append("guidance.md deve contenere la sezione 'Fonti e riferimenti'")
    if "## Verifica rapida delle regole" not in guidance:
        errors.append("guidance.md deve contenere la sezione 'Verifica rapida delle regole'")


def check_course_index(project_dir: Path, errors: list[str]) -> None:
    course = read_text(project_dir / "corso.md")

    for lesson_number in lesson_numbers_from_studio(project_dir):
        lesson = f"{lesson_number:02d}"
        if f"Lezione {lesson} -" not in course:
            errors.append(f"corso.md non contiene la voce Lezione {lesson}")
        if not re.search(rf"^## Lezione {lesson} - ", course, flags=re.MULTILINE):
            errors.append(f"corso.md non contiene la sezione ## Lezione {lesson}")


def check_study_scripts(project_dir: Path, errors: list[str]) -> None:
    for script in sorted(lessons_dir(project_dir).glob("[0-9][0-9]_*.py")):
        lesson = script.name[:2]
        text = read_text(script)

        try:
            module = ast.parse(text)
        except SyntaxError as exc:
            errors.append(f"{script}: sintassi non valida: {exc}")
            continue

        docstring = ast.get_docstring(module) or ""
        if "Differenza rispetto" not in docstring:
            errors.append(f"{script}: manca 'Differenza rispetto' nella docstring")
        if "Scopo del file" not in docstring:
            errors.append(f"{script}: manca 'Scopo del file' nella docstring")

        if lesson >= "12":
            expected = f"from studio.snapshot.lezione_{lesson}.model import LanguageModel"
            if expected not in text:
                errors.append(f"{script}: import model non coerente con lezione_{lesson}")

        dataset_path_lines = [
            line for line in text.splitlines() if line.startswith("DATASET_PATH =")
        ]
        for line in dataset_path_lines:
            if '"snapshot"' in line and f'"lezione_{lesson}"' not in line:
                errors.append(f"{script}: DATASET_PATH non punta a lezione_{lesson}")


def check_english_code_identifiers(project_dir: Path, errors: list[str]) -> None:
    for path in sorted(project_dir.glob("**/*.py")):
        text = read_text(path)

        try:
            module = ast.parse(text)
        except SyntaxError:
            continue

        old_names = sorted(collect_identifiers(module) & FORBIDDEN_ITALIAN_IDENTIFIERS)
        if old_names:
            errors.append(f"{path}: identificatori Python non inglesi: {old_names}")


def check_models(project_dir: Path, errors: list[str]) -> None:
    model_paths = sorted(snapshots_dir(project_dir).glob("lezione_*/model.py"))
    model_paths.append(project_dir / "progetto_finale" / "model.py")

    for path in model_paths:
        text = read_text(path)

        try:
            module = ast.parse(text)
        except SyntaxError as exc:
            errors.append(f"{path}: sintassi non valida: {exc}")
            continue

        classes = [node.name for node in module.body if isinstance(node, ast.ClassDef)]
        if classes.count("LanguageModel") != 1:
            errors.append(f"{path}: deve contenere esattamente una classe LanguageModel")

        old_names = sorted(set(classes) & OLD_MODEL_NAMES)
        if old_names:
            errors.append(f"{path}: contiene vecchi nomi modello: {old_names}")

        docstring = ast.get_docstring(module) or ""
        if "Differenza rispetto" not in docstring:
            errors.append(f"{path}: manca 'Differenza rispetto' nella docstring")
        if "Scopo del file" not in docstring:
            errors.append(f"{path}: manca 'Scopo del file' nella docstring")


def check_course_lesson_references(project_dir: Path, errors: list[str]) -> None:
    course = read_text(project_dir / "corso.md")
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
            r"studio/snapshot[\" /]+lezione_(\d{2})|studio\.snapshot\.lezione_(\d{2})",
            body,
        )
        found = {left or right for left, right in refs}
        wrong = sorted(number for number in found if number != lesson)
        if wrong:
            errors.append(f"Lezione {lesson}: riferimenti a snapshot diversi: {wrong}")


def check_complete_code_blocks(project_dir: Path, errors: list[str]) -> None:
    course = read_text(project_dir / "corso.md")
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
            elif raw_path.startswith(("studio/", "progetto_finale/", "strumenti/")):
                file_path = project_dir / raw_path
            else:
                continue

            if not file_path.exists():
                errors.append(f"corso.md cita un file inesistente: {raw_path}")
                continue

            actual = read_text(file_path).strip() + "\n"
            if code != actual:
                errors.append(f"corso.md non è allineato a {raw_path}")


def check_final_project_snapshot(project_dir: Path, errors: list[str]) -> None:
    final_dir = project_dir / "progetto_finale"
    snapshot_dir = snapshots_dir(project_dir) / "lezione_42"

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
            errors.append(f"lezione_42 non contiene file finali: {missing}")
        if extra:
            errors.append(f"lezione_42 contiene file non presenti nel finale: {extra}")
        return

    for filename in sorted(final_files):
        final_text = read_text(final_dir / filename)
        snapshot_text = read_text(snapshot_dir / filename)
        if final_text != snapshot_text:
            errors.append(
                "lezione_42 non è allineata a progetto_finale per "
                f"{filename}"
            )


def check_no_pycache(project_dir: Path, errors: list[str]) -> None:
    pycache_dirs = sorted(project_dir.rglob("__pycache__"))
    for path in pycache_dirs:
        errors.append(f"cartella __pycache__ presente: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida struttura, lezioni, snapshot e documentazione LearnGPT.",
    )
    parser.add_argument(
        "--require-data",
        action="store_true",
        help="Controlla anche la presenza dei dataset locali non versionati.",
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
    check_no_pycache(project_dir, errors)

    if errors:
        fail(errors)

    print("OK: struttura, corso, guidance, snapshot e script sono coerenti.")


if __name__ == "__main__":
    main()
