"""Independent scholarly discovery adapters; metadata is not claim evidence."""

from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import quote

from .models import PublicationCheck, SearchPage, Work, digest
from .network import HttpTransport, NetworkError

MAX_CANDIDATES = 100


def _text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(html.unescape(value).split())


def _markup_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    # Only syntactically valid XML tags, before decoding entities; mathematical
    # comparisons such as p < 0.05 and n > 100 are literal scientific text.
    tags = r'''</?[A-Za-z_][\w:.-]*(?:\s+[A-Za-z_][\w:.-]*\s*=\s*(?:"[^"]*"|'[^']*'))*\s*/?>'''
    return _text(re.sub(tags, " ", value))


def _doi(value: Any) -> str | None:
    value = _text(value)
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I).lower()
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def _year(value: Any) -> int | None:
    try:
        year = int(value)
        return year if 1000 <= year <= 3000 else None
    except (ValueError, TypeError):
        return None


def _list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _first(value: Any) -> str:
    return _text(value[0]) if isinstance(value, list) and value else _text(value)


def _notice_status(item: dict[str, Any]) -> str | None:
    relations = _dict(item.get("relation"))
    if item.get("is-retracted") is True or item.get("is_retracted") is True or item.get("isRetracted") is True:
        return "retracted"
    if relations.get("is-retracted-by") or relations.get("is-retracted"):
        return "retracted"
    if relations.get("is-corrected-by"):
        return "corrected"
    return None


class Source:
    name = "source"
    endpoint = ""

    def __init__(self, transport: HttpTransport | None = None, proxy_url: str | None = None,
                 *, page_size: int = 50, max_pages: int = 3):
        if not 1 <= page_size <= 100 or not 1 <= max_pages <= 20:
            raise ValueError("Invalid pagination bounds")
        self.transport = transport if transport is not None else HttpTransport(proxy_url)
        self.page_size = page_size
        self.max_pages = max_pages

    def _params(self, query: str, cursor: Any, size: int) -> dict[str, Any]:
        raise NotImplementedError

    def _fetch(self, params: dict[str, Any]) -> Any:
        return self.transport.get_json(self.endpoint, params)

    def _unpack(self, raw: Any, cursor: Any, size: int) -> tuple[list, int | None, Any]:
        raise NotImplementedError

    def _parse(self, item: Any) -> Work:
        raise NotImplementedError

    def _work(self, item: dict, *, identifier: str, title: Any, **kwargs: Any) -> Work:
        title = _text(title)
        if not title:
            raise ValueError("Missing title")
        identifier = identifier or digest(title + repr(kwargs.get("authors", [])))[:24]
        status = _notice_status(item)
        checks = [PublicationCheck(source=self.name, status=status,
                                   detail="Source metadata notice flag")] if status else []
        return Work(id=f"{self.name}:{identifier}", title=title,
                    source_ids={self.name: identifier}, **kwargs,
                    publication_checks=checks if status else [])

    def search(self, query: str, limit: int) -> SearchPage:
        limit = min(MAX_CANDIDATES, max(0, int(limit)))
        page = SearchPage(source=self.name, queries=[query])
        if limit == 0 or not query.strip():
            page.status = "empty"
            return page
        cursor: Any = None
        seen_ids: set[str] = set()
        seen_cursors: set[str] = set()
        invalid = 0
        for page_number in range(self.max_pages):
            size = min(self.page_size, limit - len(page.records))
            params = self._params(query, cursor, size)
            try:
                raw = self._fetch(params)
                page.raw_responses.append({"url": self.endpoint, "params": params, "body": raw})
                items, total, next_cursor = self._unpack(raw, cursor, size)
                total = total if isinstance(total, int) and total >= 0 else None
                page.total = total
                for index, item in enumerate(items):
                    try:
                        work = self._parse(item)
                    except (ValueError, TypeError, AttributeError, KeyError, IndexError):
                        invalid += 1
                        continue
                    if work.id not in seen_ids:
                        work.provenance.append({"source": self.name, "query": query,
                                                "raw_response_index": page_number, "item_index": index,
                                                "retrieved_at": page.retrieved_at})
                        page.records.append(work)
                        seen_ids.add(work.id)
                    if len(page.records) >= limit:
                        break
                has_more = next_cursor is not None
                if len(page.records) >= limit or page_number + 1 == self.max_pages:
                    page.truncated = has_more or len(items) > size or (
                        total is not None and total > len(page.records))
                    break
                if not has_more:
                    if total is not None and total > len(page.records):
                        page.truncated = True
                        page.errors.append("Source total exceeds returned records but continuation cursor is missing")
                    break
                if repr(next_cursor) in seen_cursors:
                    page.errors.append("Source repeated a pagination cursor")
                    page.truncated = True
                    break
                seen_cursors.add(repr(next_cursor))
                cursor = next_cursor
            except Exception as exc:
                # Never surface arbitrary client exception strings, which may contain URLs/credentials.
                page.errors.append(str(exc) if isinstance(exc, NetworkError) else
                                   f"Source response failed ({type(exc).__name__})")
                page.status = ("partial" if page.records else "rate_limited"
                               if isinstance(exc, NetworkError) and exc.rate_limited else "unavailable")
                page.truncated = bool(page.records)
                return page
        if invalid:
            page.errors.append(f"Skipped {invalid} malformed record(s)")
        page.status = "partial" if page.errors and page.records else "unavailable" if page.errors else (
            "ok" if page.records else "empty")
        return page


class OpenAlexSource(Source):
    name = "openalex"
    endpoint = "https://api.openalex.org/works"

    def _params(self, query, cursor, size):
        return {"search": query, "per-page": size, "cursor": cursor or "*"}

    def _unpack(self, raw, cursor, size):
        if not isinstance(raw.get("results"), list):
            raise NetworkError("Malformed OpenAlex response")
        meta = _dict(raw.get("meta"))
        return raw["results"], meta.get("count"), meta.get("next_cursor") if raw["results"] else None

    def _parse(self, item):
        inverted = _dict(item.get("abstract_inverted_index"))
        tokens = [(p, word) for word, positions in inverted.items() for p in _list(positions)
                  if isinstance(p, int) and p >= 0 and isinstance(word, str)]
        abstract = " ".join(word for _, word in sorted(tokens))
        location = _dict(item.get("primary_location"))
        oa = _dict(item.get("best_oa_location"))
        urls = [v for v in [location.get("landing_page_url"), item.get("doi")] if isinstance(v, str)]
        return self._work(item, identifier=_text(item.get("id")), title=item.get("title"),
                          authors=[_text(_dict(a.get("author")).get("display_name"))
                                   for a in _list(item.get("authorships")) if isinstance(a, dict)
                                   and _text(_dict(a.get("author")).get("display_name"))],
                          year=_year(item.get("publication_year")), doi=_doi(item.get("doi")),
                          abstract=abstract, urls=urls,
                          open_access_url=oa.get("pdf_url") or oa.get("landing_page_url"),
                          venue=_text(_dict(location.get("source")).get("display_name")))


class SemanticScholarSource(Source):
    name = "semantic_scholar"
    endpoint = "https://api.semanticscholar.org/graph/v1/paper/search"

    def _params(self, query, cursor, size):
        return {"query": query, "limit": size, "offset": cursor or 0,
                "fields": "title,authors,year,abstract,externalIds,url,openAccessPdf,venue,publicationTypes"}

    def _unpack(self, raw, cursor, size):
        if not isinstance(raw.get("data"), list):
            raise NetworkError("Malformed Semantic Scholar response")
        return raw["data"], raw.get("total"), raw.get("next") if raw["data"] else None

    def _parse(self, item):
        external = _dict(item.get("externalIds"))
        url = item.get("url")
        return self._work(item, identifier=_text(item.get("paperId")), title=item.get("title"),
                          authors=[_text(a.get("name")) for a in _list(item.get("authors"))
                                   if isinstance(a, dict) and _text(a.get("name"))],
                          year=_year(item.get("year")), doi=_doi(external.get("DOI")),
                          arxiv_id=_text(external.get("ArXiv")) or None,
                          abstract=_text(item.get("abstract")), urls=[url] if isinstance(url, str) else [],
                          open_access_url=_dict(item.get("openAccessPdf")).get("url"),
                          venue=_text(item.get("venue")))


class CrossrefSource(Source):
    name = "crossref"
    endpoint = "https://api.crossref.org/works"

    def _params(self, query, cursor, size):
        return {"query.bibliographic": query, "sort": "score", "order": "desc",
                "rows": size, "cursor": cursor or "*"}

    def _unpack(self, raw, cursor, size):
        message = _dict(raw.get("message"))
        if raw.get("status") != "ok" or not isinstance(message.get("items"), list):
            raise NetworkError("Malformed Crossref response")
        items = message["items"]
        return items, message.get("total-results"), message.get("next-cursor") if items else None

    def _parse(self, item):
        year = None
        for name in ("published", "published-print", "published-online", "issued"):
            dates = _list(_dict(item.get(name)).get("date-parts"))
            if dates and _list(dates[0]):
                year = _year(dates[0][0])
                if year:
                    break
        authors = []
        for a in _list(item.get("author")):
            if isinstance(a, dict):
                name = _text(a.get("name")) or " ".join(filter(None, [_text(a.get("given")),
                                                                         _text(a.get("family"))]))
                if name:
                    authors.append(name)
        doi = _doi(item.get("DOI"))
        url = item.get("URL")
        return self._work(item, identifier=doi or "", title=_first(item.get("title")), authors=authors,
                          year=year, doi=doi, abstract=_markup_text(item.get("abstract")),
                          urls=[url] if isinstance(url, str) else [], venue=_first(item.get("container-title")))


class ArxivSource(Source):
    name = "arxiv"
    endpoint = "https://export.arxiv.org/api/query"
    namespaces = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/",
                  "x": "http://arxiv.org/schemas/atom"}

    def _params(self, query, cursor, size):
        return {"search_query": "all:" + query, "start": cursor or 0, "max_results": size}

    def _fetch(self, params):
        return self.transport.get_bytes(self.endpoint, params).decode("utf-8")

    def _unpack(self, raw, cursor, size):
        if "<!DOCTYPE" in raw.upper() or "<!ENTITY" in raw.upper():
            raise NetworkError("Unsafe XML response")
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            raise NetworkError("Expected arXiv Atom XML response") from None
        if root.tag != "{http://www.w3.org/2005/Atom}feed":
            raise NetworkError("Expected arXiv Atom feed")
        entries = root.findall("a:entry", self.namespaces)
        if any("api/errors" in (e.findtext("a:id", "", self.namespaces)) for e in entries):
            raise NetworkError("arXiv returned an API error entry")
        value = root.findtext("o:totalResults", None, self.namespaces)
        total = int(value) if value is not None else None
        next_offset = (cursor or 0) + len(entries)
        more = bool(entries) and (next_offset < total if total is not None else len(entries) == size)
        return entries, total, next_offset if more else None

    def _parse(self, item):
        def value(path):
            return item.findtext(path, "", self.namespaces)
        identifier = re.sub(r"^https?://arxiv\.org/abs/", "", value("a:id"))
        if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z.-]+/\d{7})(?:v\d+)?", identifier):
            raise ValueError("Invalid arXiv identifier")
        base = re.sub(r"v\d+$", "", identifier)
        links = item.findall("a:link", self.namespaces)
        pdf = next((e.get("href") for e in links if e.get("title") == "pdf" or
                    e.get("type") == "application/pdf"), None)
        return self._work({}, identifier=identifier, title=value("a:title"),
                          authors=[_text(a.findtext("a:name", "", self.namespaces))
                                   for a in item.findall("a:author", self.namespaces)],
                          year=_year(value("a:published")[:4]), arxiv_id=identifier,
                          doi=_doi(value("x:doi")), abstract=_text(value("a:summary")),
                          urls=[value("a:id")], open_access_url=pdf, venue=value("x:journal_ref"),
                          relations=[{"type": "version_of", "target": "arxiv:" + base}]
                          if base != identifier else [])


SOURCES: dict[str, type[Source]] = {s.name: s for s in
                                 (OpenAlexSource, SemanticScholarSource, CrossrefSource, ArxivSource)}


def collect_sources(query: str, source_names: list[str], limit: int,
                    proxy_url: str | None) -> list[SearchPage]:
    """Search concurrently; results retain caller order, including unknown sources."""
    def run(name: str) -> SearchPage:
        try:
            cls = SOURCES.get(name)
            if cls is None:
                return SearchPage(source=name, status="unavailable", queries=[query], errors=["Unknown source"])
            return cls(proxy_url=proxy_url).search(query, limit)
        except Exception as exc:
            return SearchPage(source=name, status="unavailable", queries=[query],
                              errors=[f"Source unavailable ({type(exc).__name__})"])
    if not source_names:
        return []
    with ThreadPoolExecutor(max_workers=min(4, len(source_names))) as pool:
        return list(pool.map(run, source_names))


def check_publication(work: Work, transport: HttpTransport | None = None,
                      proxy_url: str | None = None) -> list[PublicationCheck]:
    """Check Crossref notice metadata; this never validates semantic support."""
    doi = _doi(work.doi)
    if not doi:
        return [PublicationCheck(source="crossref", detail="No valid DOI available for notice lookup")]
    transport = transport if transport is not None else HttpTransport(proxy_url)
    status = None
    try:
        raw = transport.get_json("https://api.crossref.org/works/" + quote(doi, safe=""))
        if raw.get("status") != "ok" or not isinstance(raw.get("message"), dict):
            raise NetworkError("Malformed Crossref publication response")
        status = _notice_status(raw["message"])
        # Crossref update-to lives on the notice record, so inspect incoming updates.
        notices = transport.get_json("https://api.crossref.org/works", {"filter": "updates:" + doi,
                                                                       "rows": 100})
        message = _dict(notices.get("message"))
        if notices.get("status") != "ok" or not isinstance(message.get("items"), list):
            raise NetworkError("Malformed Crossref notice response")
        unknown_notice = False
        for notice in message["items"]:
            updates = _dict(notice).get("update-to")
            if not isinstance(updates, list) or not updates:
                unknown_notice = True
                continue
            matched = False
            for update in updates:
                if not isinstance(update, dict) or _doi(update.get("DOI")) != doi:
                    unknown_notice = True
                    continue
                matched = True
                kind = _text(update.get("type")).lower()
                if kind in {"retraction", "withdrawal", "removal"}:
                    status = "retracted"
                elif kind in {"correction", "erratum", "corrigendum"} and status != "retracted":
                    status = "corrected"
                elif kind not in {"correction", "erratum", "corrigendum"}:
                    unknown_notice = True
            if not matched:
                unknown_notice = True
        if status != "retracted" and (unknown_notice or message.get("total-results", 0) > len(message["items"])):
            return [PublicationCheck(source="crossref", detail="Unclassified or truncated notices; status remains unknown")]
        return [PublicationCheck(source="crossref", status=status or "no_notice_found",
                                 detail="Crossref notice metadata checked; absence of a notice does not establish "
                                        "validity or semantic support")]
    except Exception as exc:
        return [PublicationCheck(source="crossref", status="retracted" if status == "retracted" else "unknown",
                                 detail="Notice lookup unavailable: " +
                                 (str(exc) if isinstance(exc, NetworkError) else type(exc).__name__))]
