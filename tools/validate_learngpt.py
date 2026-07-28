"""
Changes compared with the previous files:
- This is not a lesson file and does not add model code.
- It validates the English code course and the bilingual graphical course.
- It keeps the English and Italian graphical lessons structurally aligned.

File purpose:
- Check that `course_en.md`, both graphical courses, `study/lessons`,
  `study/snapshots`, `final_project`, `tools`, and `data` stay consistent.
- Prevent a lesson from importing project code from another lesson snapshot.
- Ensure the course stays focused on explanations and snippets instead of full
  duplicated source files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import subprocess
import tokenize
from decimal import Decimal, InvalidOperation
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
    "CUDA_TRAINING_OPTIMIZATIONS.md",
    "LICENSE",
    "README.md",
    "course_en.md",
    "course_en_graphic.md",
    "course_it_graphic.md",
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

FORBIDDEN_CURRENT_COURSE_EXPERIMENT_PATTERNS = (
    re.compile(
        r"\b124(?:\s*[-‐‑‒–—−]\s*)?\s*M\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b124(?:\s*[-‐‑‒–—−]\s*)?\s*(?:million|milioni)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bGPT[\s\-‐‑‒–—−]*2\s+Small\b",
        flags=re.IGNORECASE,
    ),
)

MERMAID_FENCE_LANGUAGES = {"mermaid", "learngpt-mermaid"}
TENSOR_DIMENSION_NAMES = "BTCVDHLN"

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


def scan_fenced_blocks(markdown: str) -> list[tuple[str, str, int, bool]]:
    """Return (language, content, opening line, closed) for Markdown fences."""
    lines = markdown.splitlines()
    blocks: list[tuple[str, str, int, bool]] = []
    opening_pattern = re.compile(
        r"^[ \t]*(?P<fence>`{3,}|~{3,})(?P<info>[^\r\n]*)$"
    )
    index = 0

    while index < len(lines):
        opening = opening_pattern.match(lines[index])
        if opening is None:
            index += 1
            continue

        fence = opening.group("fence")
        info = opening.group("info").strip()
        language = info.split(maxsplit=1)[0] if info else ""
        closing_pattern = re.compile(
            rf"^[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$"
        )
        opening_line = index + 1
        content_start = index + 1
        index = content_start
        while index < len(lines) and closing_pattern.match(lines[index]) is None:
            index += 1

        closed = index < len(lines)
        blocks.append(
            (
                language,
                "\n".join(lines[content_start:index]),
                opening_line,
                closed,
            )
        )
        index += 1 if closed else 0

    return blocks


def require_visual_object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def require_visual_list(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must be an array")
    return value


def require_non_empty_visual_list(value: object, path: str) -> list[object]:
    items = require_visual_list(value, path)
    if not items:
        raise ValueError(f"{path} must contain at least one item")
    return items


def require_visual_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


MISSING_VISUAL_FIELD = object()


def optional_visual_string(value: object, path: str) -> None:
    if value is not MISSING_VISUAL_FIELD:
        require_visual_string(value, path)


def parse_visual_json(content: str) -> object:
    def reject_non_json_constant(constant: str) -> object:
        raise ValueError(f"invalid JSON constant {constant}")

    return json.loads(content, parse_constant=reject_non_json_constant)


def validate_visual_matrix(value: object, path: str) -> None:
    matrix = require_visual_object(value, path)
    require_visual_string(matrix.get("label"), f"{path}.label")
    optional_visual_string(
        matrix.get("shape", MISSING_VISUAL_FIELD),
        f"{path}.shape",
    )
    rows = require_non_empty_visual_list(matrix.get("values"), f"{path}.values")
    parsed_rows: list[list[object]] = []
    for row_index, row in enumerate(rows):
        cells = require_non_empty_visual_list(
            row,
            f"{path}.values[{row_index}]",
        )
        for column_index, cell in enumerate(cells):
            require_visual_string(
                cell,
                f"{path}.values[{row_index}][{column_index}]",
            )
        parsed_rows.append(cells)

    width = len(parsed_rows[0])
    if any(len(row) != width for row in parsed_rows):
        raise ValueError(f"{path}.values must be rectangular")


def validate_visual_definition(value: object) -> str:
    visual = require_visual_object(value, "visual")
    visual_type = require_visual_string(visual.get("type"), "type")
    require_visual_string(visual.get("title"), "title")
    require_visual_string(visual.get("description"), "description")

    if visual_type == "matrix-operation":
        operands = require_non_empty_visual_list(
            visual.get("operands"),
            "operands",
        )
        for index, operand in enumerate(operands):
            validate_visual_matrix(operand, f"operands[{index}]")
        operators = require_visual_list(visual.get("operators"), "operators")
        for index, operator in enumerate(operators):
            require_visual_string(operator, f"operators[{index}]")
        if len(operators) != len(operands) - 1:
            raise ValueError(
                "operators must contain exactly one item between each "
                "pair of operands"
            )
        validate_visual_matrix(visual.get("result"), "result")
        return visual_type

    if visual_type == "labeled-grid":
        columns = require_non_empty_visual_list(
            visual.get("columns"),
            "columns",
        )
        for index, column in enumerate(columns):
            require_visual_string(column, f"columns[{index}]")
        rows = require_non_empty_visual_list(visual.get("rows"), "rows")
        allowed_states = {"default", "highlighted", "masked"}
        for row_index, row in enumerate(rows):
            row_object = require_visual_object(row, f"rows[{row_index}]")
            require_visual_string(
                row_object.get("label"),
                f"rows[{row_index}].label",
            )
            cells = require_visual_list(
                row_object.get("cells"),
                f"rows[{row_index}].cells",
            )
            if len(cells) != len(columns):
                raise ValueError(
                    f"rows[{row_index}].cells must contain exactly "
                    f"{len(columns)} items"
                )
            for cell_index, cell in enumerate(cells):
                cell_path = f"rows[{row_index}].cells[{cell_index}]"
                cell_object = require_visual_object(cell, cell_path)
                require_visual_string(cell_object.get("value"), f"{cell_path}.value")
                state = require_visual_string(
                    cell_object.get("state", "default"),
                    f"{cell_path}.state",
                )
                if state not in allowed_states:
                    raise ValueError(
                        f"{cell_path}.state must be default, highlighted, or masked"
                    )
        return visual_type

    if visual_type == "tensor-flow":
        stages = require_visual_list(visual.get("stages"), "stages")
        if len(stages) < 2:
            raise ValueError("stages must contain at least two items")
        for index, stage in enumerate(stages):
            stage_path = f"stages[{index}]"
            stage_object = require_visual_object(stage, stage_path)
            require_visual_string(stage_object.get("label"), f"{stage_path}.label")
            require_visual_string(stage_object.get("shape"), f"{stage_path}.shape")
            optional_visual_string(
                stage_object.get("note", MISSING_VISUAL_FIELD),
                f"{stage_path}.note",
            )
        return visual_type

    raise ValueError(
        "type must be matrix-operation, labeled-grid, or tensor-flow"
    )


def validate_visual_blocks(
    name: str,
    markdown: str,
    errors: list[str],
) -> None:
    for language, content, line_number, closed in scan_fenced_blocks(markdown):
        if language != "learngpt-visual" and language not in MERMAID_FENCE_LANGUAGES:
            continue
        if not closed:
            errors.append(
                f"{name}:{line_number} has an unclosed {language} fence"
            )
            continue
        if language in MERMAID_FENCE_LANGUAGES:
            if not content.strip():
                errors.append(
                    f"{name}:{line_number} has an empty {language} fence"
                )
            continue
        try:
            definition = parse_visual_json(content)
        except (json.JSONDecodeError, ValueError) as error:
            detail = (
                error.msg
                if isinstance(error, json.JSONDecodeError)
                else str(error)
            )
            errors.append(
                f"{name}:{line_number} contains invalid learngpt-visual JSON: "
                f"{detail}"
            )
            continue
        try:
            validate_visual_definition(definition)
        except ValueError as error:
            errors.append(
                f"{name}:{line_number} has an invalid learngpt-visual block: "
                f"{error}"
            )


def compact_visual_signature(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def canonical_visual_number(value: str) -> str:
    try:
        number = Decimal(value.replace("−", "-"))
    except InvalidOperation:
        return value
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def visual_value_invariants(value: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(value, str):
        return (), ()
    normalized = value.replace("−", "-")
    numbers = tuple(
        canonical_visual_number(number)
        for number in re.findall(r"(?<![\w.])-?(?:\d+(?:\.\d*)?|\.\d+)", normalized)
    )
    dimensions = tuple(
        re.findall(
            rf"(?<![A-Za-z0-9_])[{TENSOR_DIMENSION_NAMES}]"
            r"(?![A-Za-z0-9_])",
            normalized,
        )
    )
    if "∞" in normalized:
        numbers += ("-∞" if "-∞" in normalized else "∞",)
    return numbers, dimensions


def canonical_tensor_shape(value: object) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(value, str):
        return (), ()
    normalized = value.replace("−", "-")
    if "=" in normalized:
        right_hand_side = normalized.rsplit("=", maxsplit=1)[1]
        if len(
            re.findall(
                rf"(?<![A-Za-z0-9_])[{TENSOR_DIMENSION_NAMES}]"
                r"(?![A-Za-z0-9_])",
                right_hand_side,
            )
        ) >= 2:
            normalized = right_hand_side
        else:
            normalized = re.sub(
                rf"(?:\d+\s*)?[{TENSOR_DIMENSION_NAMES}]\s*=\s*"
                r"(-?(?:\d+(?:\.\d*)?|\.\d+))",
                r"\1",
                normalized,
            )
    numbers, dimensions = visual_value_invariants(normalized)
    return numbers, tuple(dict.fromkeys(dimensions))


def canonical_visual_operator(value: object) -> object:
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower()
    return {
        "concatena": "concat",
        "concatenate": "concat",
    }.get(normalized, normalized)


def visual_structure_signature(definition: object) -> str | None:
    if not isinstance(definition, dict):
        return None

    visual_type = definition.get("type")
    if visual_type == "matrix-operation":
        operands = definition.get("operands")
        operators = definition.get("operators")
        result = definition.get("result")
        if (
            not isinstance(operands, list)
            or not isinstance(operators, list)
            or not isinstance(result, dict)
        ):
            return None
        matrices = [*operands, result]
        dimensions: list[str] = []
        semantic_matrices: list[object] = []
        for matrix in matrices:
            if not isinstance(matrix, dict):
                return None
            values = matrix.get("values")
            if not isinstance(values, list) or not values:
                return None
            first_row = values[0]
            if not isinstance(first_row, list):
                return None
            dimensions.append(f"{len(values)}x{len(first_row)}")
            semantic_matrices.append(
                {
                    "shape": canonical_tensor_shape(matrix.get("shape")),
                    "values": values,
                }
            )
        operand_dimensions = ",".join(dimensions[:-1])
        semantic_hash = compact_visual_signature(
            {
                "operators": [
                    canonical_visual_operator(operator)
                    for operator in operators
                ],
                "matrices": semantic_matrices,
            }
        )
        return (
            f"matrix-operation:{len(operands)}:"
            f"{operand_dimensions}->{dimensions[-1]}:{semantic_hash}"
        )

    if visual_type == "labeled-grid":
        columns = definition.get("columns")
        rows = definition.get("rows")
        if not isinstance(columns, list) or not isinstance(rows, list):
            return None
        semantic_rows: list[object] = []
        for row in rows:
            if not isinstance(row, dict):
                return None
            cells = row.get("cells")
            if not isinstance(cells, list):
                return None
            semantic_cells: list[object] = []
            for cell in cells:
                if not isinstance(cell, dict):
                    return None
                semantic_cells.append(
                    {
                        "value": visual_value_invariants(cell.get("value")),
                        "state": cell.get("state", "default"),
                    }
                )
            semantic_rows.append(semantic_cells)
        semantic_hash = compact_visual_signature(semantic_rows)
        return (
            f"labeled-grid:{len(rows)}x{len(columns)}:{semantic_hash}"
        )

    if visual_type == "tensor-flow":
        stages = definition.get("stages")
        if not isinstance(stages, list):
            return None
        shapes: list[object] = []
        for stage in stages:
            if not isinstance(stage, dict):
                return None
            shapes.append(canonical_tensor_shape(stage.get("shape")))
        semantic_hash = compact_visual_signature(shapes)
        return f"tensor-flow:{len(stages)}:{semantic_hash}"

    return None


def lesson_visual_sequence(markdown: str) -> list[str]:
    sequence: list[str] = []
    for language, content, _, closed in scan_fenced_blocks(markdown):
        if language in MERMAID_FENCE_LANGUAGES:
            sequence.append(
                language if closed and content.strip() else f"{language}:invalid"
            )
        elif language == "learngpt-visual":
            if not closed:
                sequence.append("learngpt-visual:invalid")
                continue
            try:
                definition = parse_visual_json(content)
                visual_signature = visual_structure_signature(definition)
            except (json.JSONDecodeError, ValueError):
                visual_signature = None
            sequence.append(
                visual_signature
                if visual_signature is not None
                else "learngpt-visual:invalid"
            )
    return sequence


IGNORED_PYTHON_TOKEN_TYPES = {
    tokenize.ENCODING,
    tokenize.ENDMARKER,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.NEWLINE,
    tokenize.NL,
}


def python_token_sequence(source: str) -> list[tuple[int, str]]:
    sequence: list[tuple[int, str]] = []
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)

    while True:
        try:
            token = next(tokens)
        except StopIteration:
            break
        except (IndentationError, tokenize.TokenError):
            # Reference-code lines can intentionally be fragments, such as the
            # first half of a parenthesized call. Tokens produced before the
            # incomplete boundary are still suitable for literal matching.
            break
        if token.type not in IGNORED_PYTHON_TOKEN_TYPES:
            sequence.append((token.type, token.string))

    return sequence


def contains_token_sequence(
    source: list[tuple[int, str]],
    candidate: list[tuple[int, str]],
) -> bool:
    if not candidate:
        return True
    candidate_length = len(candidate)
    return any(
        source[index:index + candidate_length] == candidate
        for index in range(len(source) - candidate_length + 1)
    )


def reference_code_by_lesson(
    markdown: str,
    lesson_pattern: str,
    section_heading: str,
) -> dict[int, str]:
    headings = list(re.finditer(lesson_pattern, markdown, flags=re.MULTILINE))
    blocks: dict[int, str] = {}

    for index, heading in enumerate(headings):
        lesson_number = int(heading.group(1))
        end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(markdown)
        )
        body = markdown[heading.end():end]
        match = re.search(
            rf"^{re.escape(section_heading)}\s*$"
            rf".*?^```python\s*$\n(.*?)\n^```\s*$",
            body,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is not None:
            blocks[lesson_number] = match.group(1)

    return blocks


def check_graphic_reference_code(
    project_dir: Path,
    english: str,
    italian: str,
    errors: list[str],
) -> None:
    english_blocks = reference_code_by_lesson(
        english,
        r"^## Lesson (\d{2}) — .+$",
        "### Reference code added in this lesson",
    )
    italian_blocks = reference_code_by_lesson(
        italian,
        r"^## Lezione (\d{2}) — .+$",
        "### Codice di riferimento aggiunto in questa lezione",
    )

    expected_lessons = set(range(1, 43))
    if set(english_blocks) != expected_lessons:
        errors.append(
            "course_en_graphic.md must contain one Python reference block "
            "for every lesson"
        )
        return
    if set(italian_blocks) != expected_lessons:
        errors.append(
            "course_it_graphic.md must contain one Python reference block "
            "for every lesson"
        )
        return

    for lesson_number in sorted(expected_lessons):
        english_block = english_blocks[lesson_number]
        italian_block = italian_blocks[lesson_number]
        if english_block != italian_block:
            errors.append(
                f"Lesson {lesson_number:02d} reference code differs between "
                "English and Italian"
            )

        lesson = f"{lesson_number:02d}"
        source_paths = sorted(
            (snapshots_dir(project_dir) / f"lesson_{lesson}").glob("*.py")
        )
        source_paths.extend(
            sorted(lessons_dir(project_dir).glob(f"{lesson}_*.py"))
        )
        source_tokens = python_token_sequence(
            "\n".join(read_text(path) for path in source_paths),
        )
        mismatched_lines: list[str] = []
        for line in english_block.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            line_tokens = python_token_sequence(line)
            if not line_tokens:
                continue
            if not contains_token_sequence(source_tokens, line_tokens):
                mismatched_lines.append(stripped)

        if mismatched_lines:
            preview = ", ".join(repr(line) for line in mismatched_lines[:3])
            errors.append(
                f"Lesson {lesson} reference code is not literal current "
                f"lesson source; unmatched lines: {preview}"
            )


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
        "course_en_graphic.md",
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
        if name == "course_en.md":
            for pattern in FORBIDDEN_CURRENT_COURSE_EXPERIMENT_PATTERNS:
                if pattern.search(text):
                    errors.append(
                        "course_en.md references the separate 124M/GPT-2 "
                        "Small experiment as part of the current course"
                    )

        legacy_title = "D" + "ivina" + " " + "C" + "ommedia"
        legacy_filename = "div" + "ina" + "_" + "com" + "media"
        if legacy_title in text or legacy_filename in text:
            errors.append(f"{name} still references the old dataset")

        for reference in FORBIDDEN_PUBLIC_REFERENCES:
            if reference in text:
                errors.append(f"{name} still contains legacy reference: {reference}")

        language_scan_text = text
        if name == "course_en_graphic.md":
            # Lesson 01 deliberately uses the Unicode character "è" to show
            # why UTF-8 byte count and Python character count can differ. Keep
            # that data example without treating it as Italian prose.
            language_scan_text = language_scan_text.replace("Aè.", "")
            language_scan_text = re.sub(
                r"(?<!\w)è(?!\w)",
                "",
                language_scan_text,
            )
        if ITALIAN_PUBLIC_PATTERN.search(language_scan_text):
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
    lesson_headings = list(
        re.finditer(r"^## Lesson (\d{2}) - .+$", english_course, flags=re.MULTILINE)
    )

    for lesson_number in lesson_numbers_from_study(project_dir):
        lesson = f"{lesson_number:02d}"
        if f"Lesson {lesson} -" not in english_course:
            errors.append(f"course_en.md does not contain index item Lesson {lesson}")
        if not re.search(rf"^## Lesson {lesson} - ", english_course, flags=re.MULTILINE):
            errors.append(f"course_en.md does not contain section ## Lesson {lesson}")

    required_subsections = (
        "### Visual explanation",
        "### Code",
        "### Code syntax and logic",
    )
    maximum_syntax_points = 0
    for index, heading in enumerate(lesson_headings):
        lesson = heading.group(1)
        end = (
            lesson_headings[index + 1].start()
            if index + 1 < len(lesson_headings)
            else english_course.find("\n## Controlled Training from Scratch", heading.start())
        )
        if end == -1:
            end = len(english_course)
        body = english_course[heading.end() : end]

        subsection_positions = []
        for subsection in required_subsections:
            occurrences = [
                match.start()
                for match in re.finditer(
                    rf"^{re.escape(subsection)}$",
                    body,
                    flags=re.MULTILINE,
                )
            ]
            if len(occurrences) != 1:
                errors.append(
                    f"Lesson {lesson}: expected exactly one {subsection!r}, "
                    f"found {len(occurrences)}"
                )
            subsection_positions.append(occurrences[0] if occurrences else -1)

        if all(position >= 0 for position in subsection_positions):
            if subsection_positions != sorted(subsection_positions):
                errors.append(
                    f"Lesson {lesson}: required teaching subsections are out of order"
                )

            _, code_start, syntax_start = subsection_positions
            code_body = body[code_start:syntax_start]
            if not re.search(r"^```(?:python|bash|text)$", code_body, flags=re.MULTILINE):
                errors.append(
                    f"Lesson {lesson}: 'Code' must contain a code block"
                )

            syntax_body = body[syntax_start:]
            syntax_points = re.findall(
                r"^- .+(?:\n {2}.+)*",
                syntax_body,
                flags=re.MULTILINE,
            )
            maximum_syntax_points = max(maximum_syntax_points, len(syntax_points))
            if len(syntax_points) < 3:
                errors.append(
                    f"Lesson {lesson}: 'Code syntax and logic' must contain at least three points"
                )
            for point_index, point in enumerate(syntax_points, start=1):
                if "`" not in point:
                    errors.append(
                        f"Lesson {lesson}: syntax point {point_index} has no explicit code reference"
                    )

    if maximum_syntax_points <= 4:
        errors.append(
            "course_en.md still treats four syntax points as a fixed template"
        )


def check_graphic_course(project_dir: Path, errors: list[str]) -> None:
    english = read_text(project_dir / "course_en_graphic.md")
    italian = read_text(project_dir / "course_it_graphic.md")

    for name, markdown in (
        ("course_en_graphic.md", english),
        ("course_it_graphic.md", italian),
    ):
        for pattern in FORBIDDEN_CURRENT_COURSE_EXPERIMENT_PATTERNS:
            if pattern.search(markdown):
                errors.append(
                    f"{name} references the separate 124M/GPT-2 Small "
                    "experiment as part of the current course"
                )

    check_graphic_reference_code(project_dir, english, italian, errors)

    def check_markdown_table_widths(name: str, markdown: str) -> None:
        table_rows: list[tuple[int, str]] = []

        def flush_table() -> None:
            if len(table_rows) < 2:
                table_rows.clear()
                return
            widths = [
                len(re.findall(r"(?<!\\)\|", row))
                for _, row in table_rows
            ]
            if len(set(widths)) != 1:
                locations = ", ".join(
                    f"{line_number}:{width}"
                    for (line_number, _), width in zip(table_rows, widths)
                )
                errors.append(
                    f"{name} contains a malformed Markdown table "
                    f"(line:pipe-count {locations})"
                )
            table_rows.clear()

        for line_number, line in enumerate(markdown.splitlines(), start=1):
            if line.startswith("|") and line.endswith("|"):
                table_rows.append((line_number, line))
            else:
                flush_table()
        flush_table()

    check_markdown_table_widths("course_en_graphic.md", english)
    check_markdown_table_widths("course_it_graphic.md", italian)
    validate_visual_blocks("course_en_graphic.md", english, errors)
    validate_visual_blocks("course_it_graphic.md", italian, errors)

    structured_lessons = set(range(1, 43))
    minimum_narrative_words = {
        **{number: 180 for number in structured_lessons},
        1: 250,
        11: 450,
        19: 650,
        30: 450,
        42: 800,
    }
    expected_hash = hashlib.sha256(english.encode("utf-8")).hexdigest()
    translated_hashes = re.findall(
        r"^<!-- source-sha256: ([a-f0-9]{64}) -->$",
        italian,
        flags=re.MULTILINE,
    )
    if len(translated_hashes) != 1 or translated_hashes[0] != expected_hash:
        errors.append(
            "course_it_graphic.md must contain exactly one source-sha256 "
            "marker aligned with course_en_graphic.md"
        )

    course_contracts = (
        {
            "name": "course_en_graphic.md",
            "markdown": english,
            "lesson_pattern": r"^## Lesson (\d{2}) — .+$",
            "final_heading": "\n## Final mental model",
            "running_contract": "## Running example used by every lesson",
            "shared_sections": (
                "### Lesson summary: goal and result",
                "### How to read the mathematics",
                "### Visual worked example",
                "### Reference code added in this lesson",
                "### Syntax and logic",
            ),
            "legacy_sections": (
                "### Guided walkthrough: follow the transformation",
                "### Takeaway",
            ),
            "legacy_guide_sections": (
                "#### Follow the transformation",
                "#### Status quo after the lesson",
            ),
            "legacy_compass_fields": (
                "**Starting state:**",
                "**Transformation:**",
                "**Resulting state:**",
                "**Why now:**",
            ),
            "pilot_sections": (
                "### Understand the transformation",
                "### Transformation, step by step",
                "### Where we are now",
            ),
            "pilot_compass_fields": (
                "**Before:**",
                "**Goal:**",
                "**After:**",
                "**Invariant:**",
            ),
            "observe_label": "**What to observe:**",
            "closing_fields": (
                "**Changed:**",
                "**Preserved:**",
                "**Next:**",
                "**If you remember one thing:**",
            ),
            "notation_header": "| Notation | Read it as | Meaning here |",
            "example_marker": "**Running example state:**",
            "obsolete_sections": (
                "### Transformation trace: input → output",
                "### Status transition: before → after",
            ),
            "orientation_required": (
                "## Lesson 00 — How to use this course",
                "Quick orientation",
                "What you need before you start",
                "The companion GitHub repository",
                "github.com/ferdinandobons/learn-gpt",
                "What each part of a lesson means",
            ),
        },
        {
            "name": "course_it_graphic.md",
            "markdown": italian,
            "lesson_pattern": r"^## Lezione (\d{2}) — .+$",
            "final_heading": "\n## Modello mentale finale",
            "running_contract": "## Esempio ricorrente usato in ogni lezione",
            "shared_sections": (
                "### Sintesi della lezione: obiettivo e risultato",
                "### Come leggere la matematica",
                "### Esempio visivo svolto",
                "### Codice di riferimento aggiunto in questa lezione",
                "### Sintassi e logica",
            ),
            "legacy_sections": (
                "### Percorso guidato: segui la trasformazione",
                "### Punto chiave",
            ),
            "legacy_guide_sections": (
                "#### Segui la trasformazione",
                "#### Status quo dopo la lezione",
            ),
            "legacy_compass_fields": (
                "**Stato iniziale:**",
                "**Trasformazione:**",
                "**Stato risultante:**",
                "**Perché adesso:**",
            ),
            "pilot_sections": (
                "### Comprendere la trasformazione",
                "### Trasformazione, passo dopo passo",
                "### Dove siamo arrivati",
            ),
            "pilot_compass_fields": (
                "**Prima:**",
                "**Obiettivo:**",
                "**Dopo:**",
                "**Vincolo:**",
            ),
            "observe_label": "**Cosa osservare:**",
            "closing_fields": (
                "**Cambiato:**",
                "**Preservato:**",
                "**Prossimo passo:**",
                "**Se ricordi una sola cosa:**",
            ),
            "notation_header": "| Notazione | Leggilo come | Significato qui |",
            "example_marker": "**Stato dell'esempio ricorrente:**",
            "obsolete_sections": (
                "### Traccia della trasformazione: input → output",
                "### Stato: prima → dopo",
            ),
            "orientation_required": (
                "## Lezione 00 — Come usare questo corso",
                "Orientamento rapido",
                "Che cosa ti serve prima di iniziare",
                "Il repository GitHub collegato",
                "github.com/ferdinandobons/learn-gpt",
                "Che cosa significa ogni parte della lezione",
            ),
        },
    )

    expected_lessons = [0, *range(1, 43)]
    for contract in course_contracts:
        graphic_course = contract["markdown"]
        if contract["running_contract"] not in graphic_course:
            errors.append(
                f"{contract['name']} is missing the global running-example contract"
            )
        lesson_headings = list(
            re.finditer(
                contract["lesson_pattern"],
                graphic_course,
                flags=re.MULTILINE,
            )
        )
        found_lessons = [int(heading.group(1)) for heading in lesson_headings]
        if found_lessons != expected_lessons:
            errors.append(
                f"{contract['name']} must contain lesson 00 followed by "
                "ordered lessons 01 through 42"
            )
            continue

        for index, heading in enumerate(lesson_headings):
            lesson = heading.group(1)
            end = (
                lesson_headings[index + 1].start()
                if index + 1 < len(lesson_headings)
                else graphic_course.find(contract["final_heading"], heading.start())
            )
            if end == -1:
                end = len(graphic_course)
            body = graphic_course[heading.end():end]

            lesson_number = int(lesson)
            if lesson_number == 0:
                for marker in contract["orientation_required"]:
                    if marker not in graphic_course:
                        errors.append(
                            f"{contract['name']} lesson 00 is missing "
                            f"orientation marker {marker!r}"
                        )
                for field in contract["pilot_compass_fields"]:
                    if body.count(field) < 1:
                        errors.append(
                            f"{contract['name']} lesson 00 is missing compass "
                            f"field {field!r}"
                        )
                continue

            is_structured = lesson_number in structured_lessons
            structural_sections = (
                contract["pilot_sections"]
                if is_structured
                else contract["legacy_sections"]
            )
            if is_structured:
                ordered_sections = (
                    contract["shared_sections"][0],
                    *structural_sections,
                    *contract["shared_sections"][1:],
                )
            else:
                ordered_sections = (
                    contract["shared_sections"][0],
                    structural_sections[0],
                    *contract["shared_sections"][1:],
                    structural_sections[1],
                )
            positions = [body.find(section) for section in ordered_sections]
            for section, position in zip(
                ordered_sections,
                positions,
            ):
                if position == -1:
                    errors.append(
                        f"{contract['name']} lesson {lesson} is missing {section!r}"
                    )
            if all(position >= 0 for position in positions):
                if positions != sorted(positions):
                    errors.append(
                        f"{contract['name']} lesson {lesson} sections are out of order"
                    )
                code_start = body.find(contract["shared_sections"][3])
                syntax_start = body.find(contract["shared_sections"][4])
                code_body = body[code_start:syntax_start]
                if code_body.count("```python") != 1:
                    errors.append(
                        f"{contract['name']} lesson {lesson} must contain one "
                        "Python reference block"
                    )

            if is_structured:
                for section in contract["legacy_sections"]:
                    if section in body:
                        errors.append(
                            f"{contract['name']} lesson {lesson} retains "
                            f"legacy section {section!r}"
                        )
                for section in contract["legacy_guide_sections"]:
                    if section in body:
                        errors.append(
                            f"{contract['name']} lesson {lesson} retains "
                            f"legacy subsection {section!r}"
                        )
                pilot_steps_start = body.find(contract["pilot_sections"][1])
                pilot_steps_end = body.find(contract["pilot_sections"][2])
                pilot_steps = body[pilot_steps_start:pilot_steps_end]
                narrative_start = body.find(contract["pilot_sections"][0])
                narrative = body[narrative_start:pilot_steps_start]
                narrative_words = re.findall(
                    r"\b[\wÀ-ÖØ-öø-ÿ]+(?:[-'][\wÀ-ÖØ-öø-ÿ]+)*\b",
                    narrative,
                )
                minimum_words = minimum_narrative_words[lesson_number]
                if len(narrative_words) < minimum_words:
                    errors.append(
                        f"{contract['name']} lesson {lesson} narrative has "
                        f"{len(narrative_words)} words; expected at least "
                        f"{minimum_words}"
                    )
                step_count = len(
                    re.findall(
                        r"^\d+\. \*\*(?:INPUT|OPERATION|INTERMEDIATE STATE|"
                        r"CONSTRAINT|CHECK|OUTPUT)\b",
                        pilot_steps,
                        flags=re.MULTILINE,
                    )
                )
                if not 3 <= step_count <= 7:
                    errors.append(
                        f"{contract['name']} lesson {lesson} must contain "
                        "between three and seven semantic transformation steps"
                    )
                if pilot_steps.count(contract["observe_label"]) != step_count:
                    errors.append(
                        f"{contract['name']} lesson {lesson} must give every "
                        "transformation step one observation cue"
                    )
                for field in contract["closing_fields"]:
                    if field not in body:
                        errors.append(
                            f"{contract['name']} lesson {lesson} is missing "
                            f"closing field {field!r}"
                        )
            else:
                for section in contract["legacy_guide_sections"]:
                    if body.count(section) != 1:
                        errors.append(
                            f"{contract['name']} lesson {lesson} must contain one "
                            f"{section!r}"
                        )

            compass_fields = (
                contract["pilot_compass_fields"]
                if is_structured
                else contract["legacy_compass_fields"]
            )
            for field in compass_fields:
                if body.count(field) < 1:
                    errors.append(
                        f"{contract['name']} lesson {lesson} is missing compass "
                        f"field {field!r}"
                    )
            if contract["notation_header"] not in body:
                errors.append(
                    f"{contract['name']} lesson {lesson} is missing its local "
                    "mathematical legend"
                )
            if body.count(contract["example_marker"]) != 1:
                errors.append(
                    f"{contract['name']} lesson {lesson} must contain exactly "
                    "one running-example state"
                )
            for obsolete_section in contract["obsolete_sections"]:
                if obsolete_section in body:
                    errors.append(
                        f"{contract['name']} lesson {lesson} retains obsolete "
                        f"section {obsolete_section!r}"
                    )

        if re.search(
            r"^### (?:Question|Self-check|Quiz|Domanda|Autoverifica)\b",
            graphic_course,
            flags=re.MULTILINE | re.IGNORECASE,
        ):
            errors.append(f"{contract['name']} must not add lesson quizzes yet")

    if "## Exploded end-to-end map" not in english:
        errors.append("course_en_graphic.md must contain the exploded end-to-end map")
    exploded_map_markers = (
        'subgraph DATA["1 · Text and tokenizer"]',
        'subgraph BIGRAM["3A · Educational bigram path"]',
        'subgraph BUILD["3B · Model construction and initialization"]',
        'subgraph BLOCK["5 · One pre-norm Transformer block · repeated L times"]',
        'subgraph SAVE["10 · Best/latest checkpoint lifecycle"]',
        'subgraph GENERATE["12 · Autoregressive generation"]',
    )
    for marker in exploded_map_markers:
        if marker not in english:
            errors.append(
                f"course_en_graphic.md exploded map is missing {marker}"
            )
    english_mermaid_count = sum(
        language in MERMAID_FENCE_LANGUAGES and closed and bool(content.strip())
        for language, content, _, closed in scan_fenced_blocks(english)
    )
    italian_mermaid_count = sum(
        language in MERMAID_FENCE_LANGUAGES and closed and bool(content.strip())
        for language, content, _, closed in scan_fenced_blocks(italian)
    )
    if english_mermaid_count < 20:
        errors.append(
            "course_en_graphic.md must retain the expected overview and "
            "supporting diagrams after the readability migration"
        )
    if italian_mermaid_count != english_mermaid_count:
        errors.append(
            "course_it_graphic.md must retain the same number of Mermaid "
            "diagrams as course_en_graphic.md"
        )
    attention_start = english.find("## Lesson 19 — Causal self-attention head")
    attention_end = english.find("## Lesson 20 —", attention_start)
    attention_lesson = english[attention_start:attention_end]
    attention_contract = (
        "q_sleeps = [1 1]",
        "q_sleeps × Kᵀ = [1 1 2 2 2]",
        "softmax       ≈  [0.25 0.25 0.50 0.00 0.00]",
        "[0.25 0.25 0.50 0 0] × V ≈ [1.00 1.25]",
        "| Query position | Keys allowed by the mask | Keys excluded |",
        r"S_{bij}",
        r"O_{bid}",
    )
    for marker in attention_contract:
        if marker not in attention_lesson:
            errors.append(
                "Graphic lesson 19 is missing attention teaching marker "
                f"{marker!r}"
            )
    attention_it_start = italian.find(
        "## Lezione 19 — Causal self-attention head"
    )
    attention_it_end = italian.find("## Lezione 20 —", attention_it_start)
    attention_it_lesson = italian[attention_it_start:attention_it_end]
    attention_it_contract = (
        "q_sleeps = [1  1]",
        "q_sleeps × Kᵀ = [1  1  2  2  2]",
        "≈ [0.25  0.25  0.50   0   0]",
        "o_sleeps ≈ [1.00  1.25]",
        "| Key alla posizione $j$ | Relazione con `sleeps` |",
        r"S_{ij}",
        r"O_i",
    )
    for marker in attention_it_contract:
        if marker not in attention_it_lesson:
            errors.append(
                "Graphic Italian lesson 19 is missing attention teaching "
                f"marker {marker!r}"
            )

    lesson_alignment_contracts = (
        (
            "English lesson 01",
            english,
            "## Lesson 01 —",
            "## Lesson 02 —",
            (
                "tiny teaching file",
                "`data/study_sample.txt`",
                "[84, 104, 101, 32, 99]",
                "`Aè.`",
            ),
            (),
        ),
        (
            "Italian lesson 01",
            italian,
            "## Lezione 01 —",
            "## Lezione 02 —",
            (
                "mini-file didattico",
                "`data/study_sample.txt`",
                "[84, 104, 101, 32, 99]",
                "`Aè.`",
            ),
            (),
        ),
        (
            "English lesson 11",
            english,
            "## Lesson 11 —",
            "## Lesson 12 —",
            (
                "tensor[:, 1]",
                r"X_{b,t}",
                "No model parameter and no matrix multiplication",
            ),
            ("[[17]", "A @ b"),
        ),
        (
            "Italian lesson 11",
            italian,
            "## Lezione 11 —",
            "## Lezione 12 —",
            (
                "tensor[:, 1]",
                r"X_{b,t}",
                "non compaiono ancora",
                "né matrix multiplication",
            ),
            ("[[17]", "A @ b"),
        ),
        (
            "English lesson 30",
            english,
            "## Lesson 30 —",
            "## Lesson 31 —",
            (
                "exactly seven related fields",
                "not an atomic write",
                r"(\theta,\omega,\gamma_m,\tau,h,\phi,\phi^{-1})",
            ),
            (r"P_{\mathrm{tmp}}",),
        ),
        (
            "Italian lesson 30",
            italian,
            "## Lezione 30 —",
            "## Lezione 31 —",
            (
                "esattamente sette campi",
                "non è ancora un",
                "salvataggio atomico",
                r"(\theta,\omega,\gamma_m,\tau,h,\phi,\phi^{-1})",
            ),
            (r"P_{\mathrm{tmp}}",),
        ),
        (
            "English lesson 42",
            english,
            "## Lesson 42 —",
            "\n## Final mental model",
            (
                "`docs/FINAL_TRAINING_RUNBOOK.md`",
                r"$T=32$, $C=64$, $H=4$, and $L=2$",
                "17,716,049",
            ),
            (),
        ),
        (
            "Italian lesson 42",
            italian,
            "## Lezione 42 —",
            "\n## Modello mentale finale",
            (
                "`docs/FINAL_TRAINING_RUNBOOK.md`",
                r"$T=32$",
                r"$C=64$",
                r"$H=4$",
                r"$L=2$",
                "17.716.049",
            ),
            (),
        ),
    )
    for (
        label,
        document,
        start_marker,
        end_marker,
        required_markers,
        forbidden_markers,
    ) in lesson_alignment_contracts:
        start = document.find(start_marker)
        end = document.find(end_marker, start)
        lesson_body = document[start:end if end != -1 else len(document)]
        for marker in required_markers:
            if marker not in lesson_body:
                errors.append(f"{label} is missing alignment marker {marker!r}")
        for marker in forbidden_markers:
            if marker in lesson_body:
                errors.append(f"{label} retains contradictory marker {marker!r}")

    for number in structured_lessons:
        english_start = english.find(f"## Lesson {number:02d} —")
        english_end = english.find(
            f"## Lesson {number + 1:02d} —",
            english_start,
        )
        if number == 42 or english_end == -1:
            english_end = english.find("\n## Final mental model", english_start)
        italian_start = italian.find(f"## Lezione {number:02d} —")
        italian_end = italian.find(
            f"## Lezione {number + 1:02d} —",
            italian_start,
        )
        if number == 42 or italian_end == -1:
            italian_end = italian.find(
                "\n## Modello mentale finale",
                italian_start,
            )
        english_lesson = english[
            english_start:english_end if english_end != -1 else len(english)
        ]
        italian_lesson = italian[
            italian_start:italian_end if italian_end != -1 else len(italian)
        ]
        english_visual_sequence = lesson_visual_sequence(english_lesson)
        italian_visual_sequence = lesson_visual_sequence(italian_lesson)
        if english_visual_sequence != italian_visual_sequence:
            errors.append(
                f"Lesson {number:02d} Visual Kit sequence differs: "
                f"English {english_visual_sequence}, "
                f"Italian {italian_visual_sequence}"
            )
        english_steps = len(
            re.findall(
                r"^\d+\. \*\*(?:INPUT|OPERATION|INTERMEDIATE STATE|"
                r"CONSTRAINT|CHECK|OUTPUT)\b",
                english_lesson,
                flags=re.MULTILINE,
            )
        )
        italian_steps = len(
            re.findall(
                r"^\d+\. \*\*(?:INPUT|OPERATION|INTERMEDIATE STATE|"
                r"CONSTRAINT|CHECK|OUTPUT)\b",
                italian_lesson,
                flags=re.MULTILINE,
            )
        )
        if english_steps != italian_steps:
            errors.append(
                f"Lesson {number:02d} has {english_steps} English steps "
                f"but {italian_steps} Italian steps"
            )


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
            errors.append(f"how-to-train runbook is missing: {term}")
    for stale_term in (
        "cu128",
        "learngpt-cuda.pt",
        "--context-size 128",
        "--eval-interval 20",
    ):
        if stale_term in runbook:
            errors.append(f"how-to-train runbook contains stale profile: {stale_term}")

    video_guide = read_text(video_guide_path)
    for episode in range(1, 11):
        if f"## Episode {episode} " not in video_guide:
            errors.append(f"video series guide is missing Episode {episode}")

    try:
        workflow = json.loads(read_text(workflow_path))
    except json.JSONDecodeError as error:
        errors.append(f"invalid training workflow JSON: {error}")
        return

    if workflow.get("version") != 2:
        errors.append("training workflow version must be 2")

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
    quick_start = workflow.get("quickStart") or {}
    if [platform.get("id") for platform in quick_start.get("platforms", [])] != [
        "mps",
        "cuda",
    ]:
        errors.append("training quick start must define MPS and CUDA platforms")
    quick_steps = quick_start.get("steps") or []
    if [step.get("number") for step in quick_steps] != list(range(1, 8)):
        errors.append("training quick start must contain consecutive steps 1 through 7")
    for step in quick_steps:
        commands = step.get("commands") or {}
        for platform in ("mps", "cuda"):
            if not (commands.get(platform) or {}).get("code"):
                errors.append(
                    f"training quick-start step {step.get('number')} is missing {platform} code"
                )
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
    check_graphic_course(project_dir, errors)
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
