#!/usr/bin/env python
# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import ast
import re
import shutil
from collections import defaultdict
from pathlib import Path


try:
    import yaml
except ModuleNotFoundError as error:
    raise SystemExit(
        "PyYAML is required to prepare docs. Install docs deps (e.g. mkdocs-material and mkdocstrings[python])."
    ) from error


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "trl"
DOCS_SOURCE = ROOT / "docs" / "source"
DOCS_OUTPUT = ROOT / "docs" / "_mkdocs"
TOCTREE_PATH = DOCS_SOURCE / "_toctree.yml"
MKDOCS_CONFIG_PATH = ROOT / "mkdocs.yml"

AUTODOC_PATTERN = re.compile(r"^\s*\[\[autodoc\]\]\s+([^\s]+)\s*$")
HFOPTIONS_OPEN_PATTERN = re.compile(r"^\s*<hfoptions\b[^>]*>\s*$")
HFOPTIONS_CLOSE_PATTERN = re.compile(r"^\s*</hfoptions>\s*$")
HFOPTION_OPEN_PATTERN = re.compile(r'^\s*<hfoption\b[^>]*id="([^"]+)"[^>]*>\s*$')
HFOPTION_CLOSE_PATTERN = re.compile(r"^\s*</hfoption>\s*$")
AUTODOC_MEMBER_PATTERN = re.compile(r"^\s{4,}-\s+([^\s].*)$")

KNOWN_MODULE_PREFIXES = {
    "chat_template_utils",
    "data_utils",
    "experimental",
    "extras",
    "models",
    "rewards",
    "scripts",
    "trainer",
}

SHOW_SOURCE_SYMBOLS = {
    "trl.trainer.dar_trainer.DARTrainer",
    "trl.trainer.vmpo_trainer.VMPOTrainer",
}


def module_name_for_path(path: Path) -> str:
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def build_symbol_index() -> tuple[dict[str, list[str]], set[str]]:
    symbol_index: dict[str, list[str]] = defaultdict(list)
    all_symbols: set[str] = set()

    for file in PACKAGE_ROOT.rglob("*.py"):
        module = module_name_for_path(file)
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))

        for node in tree.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name.startswith("_"):
                continue
            symbol = f"{module}.{node.name}"
            symbol_index[node.name].append(symbol)
            all_symbols.add(symbol)

    for name, values in symbol_index.items():
        symbol_index[name] = sorted(set(values))

    return symbol_index, all_symbols


def choose_best_symbol(matches: list[str], module_hint: str | None = None) -> str:
    candidates = list(matches)

    if module_hint:
        hinted = [candidate for candidate in candidates if candidate.startswith(f"{module_hint}.")]
        if hinted:
            candidates = hinted

    non_experimental = [candidate for candidate in candidates if ".experimental." not in candidate]
    if non_experimental:
        candidates = non_experimental

    return sorted(candidates, key=lambda candidate: (candidate.count("."), len(candidate), candidate))[0]


def resolve_autodoc_target(raw_symbol: str, symbol_index: dict[str, list[str]], all_symbols: set[str]) -> str:
    symbol = raw_symbol.strip()
    if "." not in symbol:
        matches = symbol_index.get(symbol, [])
        if matches:
            return choose_best_symbol(matches, module_hint="trl")
        return f"trl.{symbol}"

    if symbol.startswith("trl."):
        candidate = symbol
    elif symbol.split(".", maxsplit=1)[0] in KNOWN_MODULE_PREFIXES:
        candidate = f"trl.{symbol}"
    else:
        candidate = symbol

    if candidate in all_symbols:
        return candidate

    module_hint, _, name = candidate.rpartition(".")
    matches = symbol_index.get(name, [])
    if matches:
        return choose_best_symbol(matches, module_hint=module_hint or None)

    return candidate


def append_blank_line(lines: list[str]) -> None:
    if lines and lines[-1] != "":
        lines.append("")


def transform_markdown(
    text: str,
    symbol_index: dict[str, list[str]],
    all_symbols: set[str],
    unresolved_symbols: set[tuple[str, str]],
) -> str:
    output_lines: list[str] = []
    lines = text.splitlines()
    line_index = 0

    while line_index < len(lines):
        line = lines[line_index]

        if HFOPTIONS_OPEN_PATTERN.match(line) or HFOPTIONS_CLOSE_PATTERN.match(line):
            line_index += 1
            continue

        hfoption_open = HFOPTION_OPEN_PATTERN.match(line)
        if hfoption_open:
            append_blank_line(output_lines)
            output_lines.append(f"### {hfoption_open.group(1).strip()}")
            output_lines.append("")
            line_index += 1
            continue

        if HFOPTION_CLOSE_PATTERN.match(line):
            append_blank_line(output_lines)
            line_index += 1
            continue

        autodoc = AUTODOC_PATTERN.match(line)
        if autodoc:
            raw_symbol = autodoc.group(1)
            resolved = resolve_autodoc_target(raw_symbol, symbol_index, all_symbols)

            members: list[str] = []
            lookahead_index = line_index + 1
            while lookahead_index < len(lines):
                member_match = AUTODOC_MEMBER_PATTERN.match(lines[lookahead_index])
                if not member_match:
                    break
                members.append(member_match.group(1).strip())
                lookahead_index += 1

            append_blank_line(output_lines)
            if resolved in all_symbols:
                output_lines.append(f"::: {resolved}")
                show_source = resolved in SHOW_SOURCE_SYMBOLS
                if members or show_source:
                    output_lines.append("    options:")
                    if show_source:
                        output_lines.append("      show_source: true")
                    output_lines.append("      members:")
                    if members:
                        for member in members:
                            output_lines.append(f"        - {member}")
                    else:
                        output_lines.append("        []")
            else:
                unresolved_symbols.add((raw_symbol, resolved))
                output_lines.append(f"### `{raw_symbol}`")
                output_lines.append("")
                output_lines.append("Reference available in source code.")
            output_lines.append("")
            line_index = lookahead_index
            continue

        output_lines.append(line)
        line_index += 1

    transformed = "\n".join(output_lines).rstrip()
    return f"{transformed}\n"


def copy_and_transform_docs(symbol_index: dict[str, list[str]], all_symbols: set[str]) -> set[tuple[str, str]]:
    unresolved_symbols: set[tuple[str, str]] = set()
    shutil.rmtree(DOCS_OUTPUT, ignore_errors=True)
    DOCS_OUTPUT.mkdir(parents=True, exist_ok=True)

    for source_file in DOCS_SOURCE.rglob("*"):
        if source_file.is_dir():
            continue

        relative_path = source_file.relative_to(DOCS_SOURCE)
        if relative_path == Path("_toctree.yml"):
            continue

        destination_file = DOCS_OUTPUT / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)

        if source_file.suffix == ".md":
            transformed = transform_markdown(
                source_file.read_text(encoding="utf-8"), symbol_index, all_symbols, unresolved_symbols
            )
            destination_file.write_text(transformed, encoding="utf-8")
        else:
            shutil.copy2(source_file, destination_file)

    return unresolved_symbols


def convert_section_to_nav(section: dict) -> dict:
    if "local" in section:
        title = section.get("title", section["local"])
        return {title: f"{section['local']}.md"}

    if "sections" in section:
        title = section.get("title", "Section")
        children = [convert_section_to_nav(child) for child in section["sections"]]
        return {title: children}

    raise ValueError(f"Unsupported toctree node: {section}")


def build_nav() -> list[dict]:
    toctree = yaml.safe_load(TOCTREE_PATH.read_text(encoding="utf-8"))
    if not isinstance(toctree, list):
        raise ValueError("Expected docs/source/_toctree.yml to contain a list.")

    nav: list[dict] = []
    for section in toctree:
        title = section.get("title")
        children = section.get("sections")
        if not title or children is None:
            raise ValueError(f"Invalid top-level toctree entry: {section}")
        nav.append({title: [convert_section_to_nav(child) for child in children]})

    return nav


def write_mkdocs_config(nav: list[dict]) -> None:
    config = {
        "site_name": "TRL Documentation",
        "site_description": "Transformer Reinforcement Learning docs",
        "docs_dir": str(DOCS_OUTPUT.relative_to(ROOT)),
        "site_dir": "site",
        "use_directory_urls": True,
        "theme": {
            "name": "material",
            "features": [
                "navigation.sections",
                "navigation.expand",
                "content.code.copy",
            ],
        },
        "plugins": [
            "search",
            {
                "mkdocstrings": {
                    "handlers": {
                        "python": {
                            "paths": ["."],
                            "options": {
                                "show_root_heading": True,
                                "show_source": False,
                            },
                        }
                    }
                }
            },
        ],
        "markdown_extensions": [
            "admonition",
            "attr_list",
            "def_list",
            "md_in_html",
            "tables",
            "toc",
            "pymdownx.highlight",
            "pymdownx.inlinehilite",
            "pymdownx.superfences",
            {
                "pymdownx.arithmatex": {
                    "generic": True,
                }
            },
        ],
        "extra_javascript": [
            "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js",
            "assets/javascripts/latex-code-comments.js",
        ],
        "extra_css": [
            "assets/stylesheets/latex-code-comments.css",
        ],
        "nav": nav,
    }

    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=120)
    MKDOCS_CONFIG_PATH.write_text(rendered, encoding="utf-8")


def main() -> None:
    symbol_index, all_symbols = build_symbol_index()
    unresolved_symbols = copy_and_transform_docs(symbol_index, all_symbols)
    nav = build_nav()
    write_mkdocs_config(nav)

    print(f"Prepared docs in {DOCS_OUTPUT.relative_to(ROOT)}")
    print(f"Wrote {MKDOCS_CONFIG_PATH.relative_to(ROOT)}")

    if unresolved_symbols:
        print("Could not resolve these autodoc entries:")
        for raw_symbol, resolved in sorted(unresolved_symbols):
            print(f"  - {raw_symbol} -> {resolved}")


if __name__ == "__main__":
    main()
