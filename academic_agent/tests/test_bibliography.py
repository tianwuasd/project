import pytest

from academic_agent.bibliography import export_bibtex, import_bibtex, import_ris, normalize_doi
from academic_agent.models import Work


def test_doi_normalization():
    assert normalize_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"
    assert normalize_doi(" DOI: 10.1234/Test ") == "10.1234/test"
    assert normalize_doi("probably a paper") is None


def test_ris_multiline_missing_doi_and_authors():
    papers = import_ris("TY  - JOUR\nTI  - A study\nAU  - Li, A\nAU  - Wang, B\nAB  - First line\n      continued\nPY  - 2024/01/02\nER  -\n")
    assert len(papers) == 1
    assert papers[0].authors == ["Li, A", "Wang, B"]
    assert papers[0].abstract == "First line continued"
    assert papers[0].doi is None and papers[0].year == 2024


def test_bibtex_roundtrip_and_nested_braces():
    work = Work(title="Fish & chips 50%", authors=["Li, A", "Wang, B"], year=2024,
                doi="10.1234/a", citekey="ref_123")
    result = import_bibtex(export_bibtex([work]))[0]
    assert (result.title, result.authors, result.doi) == (work.title, work.authors, work.doi)
    assert import_bibtex('@article{a,title={A {nested} title},year=2023}')[0].year == 2023


@pytest.mark.parametrize("text", ['@article{a,title={Unclosed}', '@string{x={Name}}', '@article{a,title=unknown}', '@article{a,year=2024}'])
def test_malformed_or_unsupported_bibtex_explicitly_rejected(text):
    with pytest.raises(ValueError):
        import_bibtex(text)


def test_missing_ris_title_and_unsafe_export_key_rejected():
    with pytest.raises(ValueError, match="title"):
        import_ris("TY  - JOUR\nER  -")
    with pytest.raises(ValueError, match="citation key"):
        export_bibtex([Work(title="Study", citekey="x},\\input{attack}")])


@pytest.mark.parametrize("value", [r"C# examples", r"A {nested {deeper}} title", r"Cost $5",
                                  r"A \\ B", r"50% & x_y ~ z^2", r"literal \textbackslash{}"])
def test_all_exported_escapes_roundtrip_repeatedly(value):
    work = Work(title=value, authors=[value], venue=value, citekey="ref_1")
    for _ in range(3):
        result = import_bibtex(export_bibtex([work]))[0]
        assert (result.title, result.authors, result.venue) == (value, [value], value)
        result.citekey = work.citekey
        work = result


@pytest.mark.parametrize("text", [r"@article{a,title={A \emph{nested} macro}}",
                                  '@article{a,title={A} # {B}}'])
def test_complex_macros_and_concatenation_rejected(text):
    with pytest.raises(ValueError, match="unsupported|Unsupported"):
        import_bibtex(text)
