"""Conservative RIS/BibTeX import and escaped BibTeX export.

BibTeX supports literal braced/quoted values and numeric values. Common character
escapes round-trip; string expansion, concatenation, and arbitrary TeX macros
are intentionally unsupported and must be expanded before import.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Work


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"^(https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value.strip(), flags=re.I)
    value = value.strip().lower()
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def _year(value: str) -> int | None:
    match = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", value)
    return int(match[0]) if match else None


def import_ris(text: str) -> list[Work]:
    records: list[Work] = []
    fields: dict[str, list[str]] = {}
    last = ""
    for line in text.splitlines() + ["ER  -"]:
        match = re.match(r"^([A-Z0-9]{2})\s{2}-\s?(.*)$", line)
        if not match:
            if line.strip() and last:
                fields[last][-1] += " " + line.strip()
            continue
        key, value = match.groups()
        if key == "ER":
            if fields:
                title = (fields.get("TI") or fields.get("T1") or [""])[0]
                if not title:
                    raise ValueError("RIS record is missing its title")
                records.append(Work(
                    title=title,
                    authors=fields.get("AU", fields.get("A1", [])),
                    year=_year((fields.get("PY") or fields.get("Y1") or [""])[0]),
                    doi=normalize_doi((fields.get("DO") or [""])[0]),
                    abstract=(fields.get("AB") or [""])[0],
                    venue=(fields.get("JO") or fields.get("T2") or [""])[0],
                    urls=fields.get("UR", []),
                    provenance=[{"source": "ris_import"}],
                ))
            fields, last = {}, ""
        else:
            fields.setdefault(key, []).append(value)
            last = key
    return records


def _braced(text: str, pos: int) -> tuple[str, int]:
    depth, start = 1, pos + 1
    pos += 1
    while pos < len(text):
        if text[pos] == "\\":
            pos += 2
            continue
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[start:pos], pos + 1
        pos += 1
    raise ValueError("Unclosed BibTeX brace")


def import_bibtex(text: str) -> list[Work]:
    records: list[Work] = []
    pos = 0
    while match := re.search(r"@(\w+)\s*\{", text[pos:]):
        kind = match[1].lower()
        opening = pos + match.end() - 1
        body, pos = _braced(text, opening)
        if kind in {"comment", "preamble"}:
            continue
        if kind == "string":
            raise ValueError("BibTeX string macros are unsupported; expand them before import")
        if "," not in body:
            raise ValueError("BibTeX record has no fields")
        original_key, body = body.split(",", 1)
        fields: dict[str, str] = {}
        offset = 0
        while offset < len(body):
            field = re.match(r"[\s,]*(\w+)\s*=\s*", body[offset:])
            if not field:
                if body[offset:].strip(" ,\r\n\t"):
                    raise ValueError("Unsupported or malformed BibTeX field")
                break
            key = field[1].lower()
            offset += field.end()
            if offset >= len(body):
                raise ValueError("Missing BibTeX value")
            if body[offset] == "{":
                value, offset = _braced(body, offset)
            elif body[offset] == '"':
                quoted = re.match(r'"((?:\\.|[^"\\])*)"', body[offset:], re.S)
                if not quoted:
                    raise ValueError("Unclosed BibTeX quote")
                value, offset = quoted[1], offset + quoted.end()
            else:
                token = re.match(r"[^,\s]+", body[offset:])
                if not token:
                    raise ValueError("Invalid BibTeX bare value")
                value, offset = token[0], offset + token.end()
                if not value.isdigit():
                    raise ValueError("BibTeX bare macros are unsupported; use braced values")
            fields[key] = _unescape(value)
        if not fields.get("title"):
            raise ValueError("BibTeX record is missing its title")
        records.append(Work(
            title=fields["title"],
            authors=[a.strip() for a in re.split(r"\s+and\s+", fields.get("author", "")) if a.strip()],
            year=_year(fields.get("year", "")),
            doi=normalize_doi(fields.get("doi")),
            abstract=fields.get("abstract", ""),
            venue=fields.get("journal", fields.get("booktitle", "")),
            urls=[fields["url"]] if fields.get("url") else [],
            provenance=[{"source": "bibtex_import", "original_key": original_key.strip()}],
        ))
    if text.strip() and not records:
        raise ValueError("No supported BibTeX records found")
    return records


def import_file(path: Path) -> list[Work]:
    if path.stat().st_size > 10 * 1024 * 1024:
        raise ValueError("Bibliography exceeds 10 MiB")
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".ris":
        records = import_ris(text)
    elif path.suffix.lower() in {".bib", ".bibtex"}:
        records = import_bibtex(text)
    else:
        raise ValueError("Bibliography must be RIS or BibTeX")
    for work in records:
        work.provenance.append({"source": "local_import", "file": path.name})
    return records


_ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "%": r"\%", "&": r"\&",
            "_": r"\_", "#": r"\#", "$": r"\$", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _unescape(value: str) -> str:
    reverse = {escaped: char for char, escaped in _ESCAPES.items()}
    # Decode once, so a literal backslash followed by TeX-looking text stays literal.
    def decode(match: re.Match) -> str:
        if match[0] not in reverse:
            raise ValueError("Complex BibTeX/TeX macros are unsupported; expand them before import")
        return reverse[match[0]]
    return re.sub(r"\\(?:[A-Za-z]+(?:\{\})?|.)", decode, value)


def _escape(value: str) -> str:
    return "".join(_ESCAPES.get(char, char) for char in value.replace("\n", " "))


def export_bibtex(works: list[Work]) -> str:
    entries = []
    for work in works:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", work.citekey):
            raise ValueError("Unsafe or missing citation key")
        fields = {"title": work.title, "author": " and ".join(work.authors), "year": str(work.year or ""),
                  "doi": work.doi or "", "journal": work.venue, "url": work.urls[0] if work.urls else ""}
        lines = [f"  {key} = {{{_escape(value)}}}" for key, value in fields.items() if value]
        entries.append("@article{" + work.citekey + ",\n" + ",\n".join(lines) + "\n}")
    return "\n\n".join(entries) + "\n"
