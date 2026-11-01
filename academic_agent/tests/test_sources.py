import socket

import httpx
import pytest

from academic_agent.models import Work
from academic_agent.network import HttpTransport, NetworkError, redact_url, route_for_url
from academic_agent.sources import (
    SOURCES,
    ArxivSource,
    CrossrefSource,
    OpenAlexSource,
    SemanticScholarSource,
    check_publication,
    collect_sources,
)


def public_resolver(host, port, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


def transport(handler, **kwargs):
    return HttpTransport(transport=httpx.MockTransport(handler), resolver=public_resolver,
                         sleep=lambda _: None, **kwargs)


def test_openalex_nested_authors_missing_doi_and_retraction():
    def handler(request):
        return httpx.Response(200, json={"results": [{
            "id": "https://openalex.org/W1", "title": "Study", "publication_year": 2024,
            "authorships": [{"author": {"display_name": "Ada Lovelace"}}, {"author": None}],
            "abstract_inverted_index": {"research": [1], "Careful": [0]}, "is_retracted": True,
        }, {"id": "https://openalex.org/W2", "title": "Metadata only"}], "meta": {"count": 2}})
    page = OpenAlexSource(transport(handler)).search("research", 10)
    assert page.status == "ok"
    assert page.records[0].authors == ["Ada Lovelace"]
    assert page.records[0].abstract == "Careful research"
    assert page.records[0].doi is None and page.records[0].retracted
    assert page.records[1].abstract == ""
    assert page.records[0].id.startswith("openalex:")
    assert page.queries == ["research"]
    assert page.raw_responses[0]["body"]["results"][0]["is_retracted"] is True
    assert page.records[0].provenance[0]["raw_response_index"] == 0


def test_semantic_scholar_pagination_and_normalization():
    offsets = []
    def handler(request):
        offset = int(request.url.params["offset"])
        offsets.append(offset)
        return httpx.Response(200, json={"data": [{"paperId": str(offset), "title": "Paper " + str(offset),
                                                  "authors": [{"name": "Jane Doe"}],
                                                  "externalIds": {"DOI": "https://doi.org/10.1234/ABC"}}],
                                         "total": 4, "next": offset + 1})
    page = SemanticScholarSource(transport(handler), page_size=1, max_pages=2).search("a", 4)
    assert offsets == [0, 1]
    assert len(page.records) == 2 and page.truncated
    assert page.status == "ok"
    assert page.records[0].doi == "10.1234/abc"
    assert page.records[0].abstract == ""


def test_crossref_author_date_abstract_and_notice():
    response = {"status": "ok", "message": {"total-results": 1, "items": [{
        "DOI": "10.1234/ABC", "title": ["Useful work"], "author": [{"given": "Jane", "family": "Doe"},
                                                                 {"name": "Research Consortium"}],
        "published-online": {"date-parts": [[2023, 1, 2]]},
        "abstract": "<jats:p>Some &amp; more</jats:p>", "relation": {"is-retracted-by": [{"id": "notice"}]},
        "link": [{"URL": "https://publisher.org/subscription.pdf", "content-type": "application/pdf"}],
    }]}}
    page = CrossrefSource(transport(lambda _: httpx.Response(200, json=response))).search("a", 5)
    work = page.records[0]
    assert work.authors == ["Jane Doe", "Research Consortium"] and work.year == 2023
    assert work.abstract == "Some & more" and work.retracted
    assert work.open_access_url is None  # Crossref links do not themselves establish open access.


def test_arxiv_preserves_version_and_pdf():
    xml = '''<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:o="http://a9.com/-/spec/opensearch/1.1/" xmlns:x="http://arxiv.org/schemas/atom">
      <o:totalResults>1</o:totalResults><entry><id>http://arxiv.org/abs/2401.12345v3</id>
      <title>A versioned paper</title><summary>Abstract only.</summary><published>2024-01-04</published>
      <author><name>Alice</name></author><author><name>Bob</name></author>
      <link title="pdf" href="https://arxiv.org/pdf/2401.12345v3" />
      <x:doi>10.1234/versioned</x:doi></entry></feed>'''
    page = ArxivSource(transport(lambda _: httpx.Response(200, content=xml))).search("paper", 10)
    work = page.records[0]
    assert page.status == "ok" and page.total == 1
    assert work.arxiv_id == "2401.12345v3" and work.id == "arxiv:2401.12345v3"
    assert work.relations == [{"type": "version_of", "target": "arxiv:2401.12345"}]
    assert work.open_access_url.endswith("v3") and work.authors == ["Alice", "Bob"]
    assert page.raw_responses[0]["body"] == xml


def test_second_page_failure_is_partial_and_preserves_raw():
    def handler(request):
        if request.url.params["cursor"] == "*":
            return httpx.Response(200, json={"results": [{"id": "W1", "title": "one"}],
                                             "meta": {"count": 3, "next_cursor": "two"}})
        return httpx.Response(503)
    page = OpenAlexSource(transport(handler)).search("paper", 3)
    assert page.status == "partial" and len(page.records) == 1
    assert len(page.raw_responses) == 1 and page.truncated
    assert "503" in page.errors[0]


def test_rate_limit_retry_and_capped_delay():
    calls, delays = [], []
    def handler(request):
        calls.append(request)
        return (httpx.Response(429, headers={"Retry-After": "99999"}) if len(calls) < 3 else
                httpx.Response(200, json={"data": [], "total": 0}))
    client = transport(handler, max_retry_after=0.25)
    client.sleep = delays.append
    page = SemanticScholarSource(client).search("paper", 10)
    assert page.status == "empty" and len(calls) == 3
    assert delays == [0.25, 0.25]


def test_exhausted_rate_limit_distinct_from_unavailable():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(429)
    page = OpenAlexSource(transport(handler)).search("paper", 3)
    assert page.status == "rate_limited" and len(calls) == 3


@pytest.mark.parametrize("cls", [OpenAlexSource, SemanticScholarSource, CrossrefSource, ArxivSource])
def test_html_error_response(cls):
    page = cls(transport(lambda _: httpx.Response(200, content="<html>Service unavailable</html>"))).search("a", 2)
    assert page.status == "unavailable" and not page.records


def test_malformed_record_skipped_but_valid_records_survive():
    raw = {"results": [{"id": "bad", "title": ""}, {"id": "good", "title": "Valid"}], "meta": {}}
    page = OpenAlexSource(transport(lambda _: httpx.Response(200, json=raw))).search("a", 3)
    assert page.status == "partial" and len(page.records) == 1
    assert "Skipped 1" in page.errors[0]


def test_collection_independent_and_deterministic(monkeypatch):
    class Good(OpenAlexSource):
        def __init__(self, **kwargs):
            super().__init__(transport(lambda _: httpx.Response(200, json={"results": [{"title": "ok"}],
                                                                          "meta": {}})))
    class Bad:
        def __init__(self, **kwargs):
            raise RuntimeError("token=do-not-surface")
    monkeypatch.setitem(SOURCES, "openalex", Good)
    monkeypatch.setitem(SOURCES, "crossref", Bad)
    pages = collect_sources("q", ["crossref", "openalex", "unknown"], 5, None)
    assert [p.source for p in pages] == ["crossref", "openalex", "unknown"]
    assert [p.status for p in pages] == ["unavailable", "ok", "unavailable"]
    assert "do-not-surface" not in str(pages)


@pytest.mark.parametrize("url,route", [("https://api.crossref.org", "proxy"),
                                       ("https://sub.cnki.net", "direct"),
                                       ("https://www.example.cn", "direct"),
                                       ("http://localhost:8000", "direct"),
                                       ("https://baidu.com.attacker.org", "proxy")])
def test_routing(url, route):
    assert route_for_url(url) == route


@pytest.mark.parametrize("url", ["http://127.0.0.1/", "http://10.0.0.1/", "http://[::1]/",
                                 "http://169.254.169.254/", "http://localhost/", "file:///etc/passwd"])
def test_private_and_invalid_targets_blocked(url):
    with pytest.raises(NetworkError):
        transport(lambda _: pytest.fail("Must not dispatch")).get_bytes(url)


def test_dns_private_address_and_redirect_are_blocked():
    client = transport(lambda _: pytest.fail("Must not dispatch"))
    client.resolver = lambda *args, **kwargs: [(2, 1, 6, "", ("192.168.1.1", 443))]
    with pytest.raises(NetworkError, match="non-public"):
        client.get_bytes("https://public.example.org/")
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(302, headers={"location": "https://127.0.0.1/admin"})
    with pytest.raises(NetworkError, match="non-public"):
        transport(handler).get_bytes("https://public.example.org/")
    assert len(calls) == 1


def test_trusted_local_mock_can_be_explicitly_allowed():
    client = transport(lambda _: httpx.Response(200, content=b"ok"), allow_private=True)
    assert client.get_bytes("http://127.0.0.1:8000/") == b"ok"


def test_redirect_cap_and_https_downgrade():
    client = transport(lambda _: httpx.Response(302, headers={"location": "/again"}), max_redirects=1)
    with pytest.raises(NetworkError, match="Redirect limit"):
        client.get_bytes("https://example.org/start")
    client = transport(lambda _: httpx.Response(302, headers={"location": "http://example.org/"}))
    with pytest.raises(NetworkError, match="Insecure redirect"):
        client.get_bytes("https://example.org/start")


def test_response_size_and_credentials_never_in_errors():
    client = transport(lambda _: httpx.Response(200, content=b"x" * 11), max_bytes=10)
    with pytest.raises(NetworkError, match="byte limit"):
        client.get_bytes("https://example.org/?api_key=SECRET")
    def handler(request):
        raise httpx.ConnectError("secret URL: " + str(request.url))
    client = transport(handler)
    with pytest.raises(NetworkError) as exc:
        client.get_json("https://example.org/?api_key=SECRET")
    assert "SECRET" not in str(exc.value)
    redacted = redact_url("https://user:pass@example.org/path?api_key=SECRET&q=paper#fragment")
    assert "SECRET" not in redacted and "pass" not in redacted and "q=paper" in redacted


def test_foreign_route_requires_explicit_proxy():
    with pytest.raises(NetworkError, match="explicit proxy"):
        HttpTransport(resolver=public_resolver).get_bytes("https://example.org/")


def test_redirect_reclassifies_route_and_ignores_environment(monkeypatch):
    real_client = httpx.Client
    configurations = []
    def handler(request):
        if request.url.host == "example.org":
            return httpx.Response(302, headers={"location": "https://example.cn/final"})
        return httpx.Response(200, content=b"done")
    def mock_client(**kwargs):
        configurations.append(dict(kwargs))
        kwargs.pop("proxy", None)
        return real_client(**kwargs, transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "Client", mock_client)
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted-environment.invalid:99")
    client = HttpTransport("http://127.0.0.1:7897", resolver=public_resolver)
    assert client.get_bytes("https://example.org/start") == b"done"
    assert configurations[0]["proxy"] == "http://127.0.0.1:7897"
    assert "proxy" not in configurations[1]
    assert all(c["trust_env"] is False and c["follow_redirects"] is False for c in configurations)
    assert all(c.get("verify", True) is True and c["timeout"] == 20 for c in configurations)


def test_candidate_cap():
    calls = []
    def handler(request):
        calls.append(request)
        offset = len(calls) * 100
        items = [{"id": str(offset + i), "title": str(offset + i)} for i in range(50)]
        return httpx.Response(200, json={"results": items, "meta": {"count": 1000, "next_cursor": "same"}})
    page = OpenAlexSource(transport(handler)).search("a", 1000)
    assert len(page.records) == 100 and len(calls) == 2 and page.truncated


def test_repeated_cursor_is_partial_instead_of_looping():
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"results": [{"id": str(len(calls)), "title": "Paper"}],
                                         "meta": {"count": 1000, "next_cursor": "same"}})
    page = OpenAlexSource(transport(handler), page_size=1).search("a", 10)
    assert len(calls) == 2 and page.status == "partial" and page.truncated


@pytest.mark.parametrize("kind,expected", [("retraction", "retracted"), ("correction", "corrected"),
                                          (None, "no_notice_found")])
def test_publication_notice_lookup(kind, expected):
    def handler(request):
        if "/works/" in request.url.path:
            return httpx.Response(200, json={"status": "ok", "message": {"DOI": "10.1234/paper"}})
        assert request.url.params["filter"] == "updates:10.1234/paper"
        items = [{"update-to": [{"DOI": "10.1234/PAPER", "type": kind}]}] if kind else []
        return httpx.Response(200, json={"status": "ok", "message": {"items": items, "total-results": len(items)}})
    checks = check_publication(Work(title="paper", doi="10.1234/paper"), transport(handler))
    assert checks[0].status == expected
    assert "semantic support" in checks[0].detail


def test_publication_unknown_when_no_doi_or_unreachable():
    client = transport(lambda _: httpx.Response(503))
    assert check_publication(Work(title="paper"), client)[0].status == "unknown"
    assert check_publication(Work(title="paper", doi="10.1234/paper"), client)[0].status == "unknown"


def test_found_retraction_survives_failure_of_incoming_notice_search():
    def handler(request):
        if "/works/" in request.url.path:
            return httpx.Response(200, json={"status": "ok", "message": {"relation": {"is-retracted-by": [1]}}})
        return httpx.Response(503)
    checks = check_publication(Work(title="paper", doi="10.1234/paper"), transport(handler))
    assert checks[0].status == "retracted"


def test_expression_of_concern_remains_unknown():
    def handler(request):
        message = ({"DOI": "10.1234/paper"} if "/works/" in request.url.path else
                   {"items": [{"update-to": [{"DOI": "10.1234/paper", "type": "expression_of_concern"}]}],
                    "total-results": 1})
        return httpx.Response(200, json={"status": "ok", "message": message})
    assert check_publication(Work(title="paper", doi="10.1234/paper"), transport(handler))[0].status == "unknown"


def test_plaintext_scientific_comparisons_are_preserved():
    abstract = "The effect held for p < 0.05 and n > 100; also p<0.05 and n>100."
    raw = {"data": [{"paperId": "1", "title": "Study", "abstract": abstract}], "total": 1}
    page = SemanticScholarSource(transport(lambda _: httpx.Response(200, json=raw))).search("q", 10)
    assert page.records[0].abstract == abstract
    xml = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/2401.12345v1</id><title>Study</title>
        <summary>''' + abstract.replace("<", "&lt;").replace(">", "&gt;") + '''</summary></entry></feed>'''
    page = ArxivSource(transport(lambda _: httpx.Response(200, content=xml))).search("q", 10)
    assert page.records[0].abstract == abstract


def test_crossref_xml_markup_keeps_comparisons_and_relevance_params():
    def handler(request):
        assert request.url.params["query.bibliographic"] == "Graph Neural Network"
        assert request.url.params["sort"] == "score" and request.url.params["order"] == "desc"
        assert "query" not in request.url.params
        return httpx.Response(200, json={"status": "ok", "message": {"total-results": 1, "items": [{
            "title": ["Study"], "abstract": '<jats:p id="s1">p<0.05 and n>100; '
            '<italic>x</italic> &lt; y and n &gt; 100.</jats:p>'}]}})
    page = CrossrefSource(transport(handler)).search("Graph Neural Network", 10)
    assert page.records[0].abstract == "p<0.05 and n>100; x < y and n > 100."


@pytest.mark.parametrize("cls,raw", [
    (OpenAlexSource, {"results": [{"id": "1", "title": "One"}], "meta": {"count": 3}}),
    (SemanticScholarSource, {"data": [{"paperId": "1", "title": "One"}], "total": 3}),
])
def test_missing_cursor_with_remaining_total_is_partial(cls, raw):
    page = cls(transport(lambda _: httpx.Response(200, json=raw))).search("q", 10)
    assert page.status == "partial" and page.truncated and len(page.records) == 1
    assert "cursor is missing" in page.errors[0]


@pytest.mark.parametrize("notice", [None, {}, {"update-to": {}}, {"update-to": []},
                                     {"update-to": {"DOI": "10.1234/paper", "type": "retraction"}},
                                     {"update-to": [None]}, {"update-to": [{"DOI": "invalid"}]},
                                     {"update-to": [{"DOI": "10.1234/paper", "type": "unknown"}]}])
@pytest.mark.parametrize("known_retracted", [False, True])
def test_unreadable_notices_are_unknown_unless_retraction_known(notice, known_retracted):
    def handler(request):
        message = ({"DOI": "10.1234/paper", "is-retracted": known_retracted}
                   if "/works/" in request.url.path else {"items": [notice], "total-results": 1})
        return httpx.Response(200, json={"status": "ok", "message": message})
    result = check_publication(Work(title="Study", doi="10.1234/paper"), transport(handler))
    assert result[0].status == ("retracted" if known_retracted else "unknown")


def test_malformed_notice_response_does_not_overstate_known_correction():
    def handler(request):
        message = ({"relation": {"is-corrected-by": ["notice"]}}
                   if "/works/" in request.url.path else {"items": None})
        return httpx.Response(200, json={"status": "ok", "message": message})
    result = check_publication(Work(title="Study", doi="10.1234/paper"), transport(handler))
    assert result[0].status == "unknown"
