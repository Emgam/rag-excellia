from pathlib import Path

from ingestion.extractors.code_extractor import extract_code_units


def test_python_function_boundaries(tmp_path: Path):
    src = (
        "def foo():\n"
        "    return 1\n"
        "\n"
        "\n"
        "class Bar:\n"
        "    def method(self):\n"
        "        return 2\n"
    )
    f = tmp_path / "sample.py"
    f.write_text(src)
    units = extract_code_units(f)
    names = {u.name for u in units}
    assert "foo" in names
    assert "Bar" in names
    foo_unit = next(u for u in units if u.name == "foo")
    assert "return 1" in foo_unit.text
    assert "class Bar" not in foo_unit.text


def test_js_brace_matching(tmp_path: Path):
    src = (
        "function add(a, b) {\n"
        "    if (a > 0) {\n"
        "        return a + b;\n"
        "    }\n"
        "    return b;\n"
        "}\n"
    )
    f = tmp_path / "sample.js"
    f.write_text(src)
    units = extract_code_units(f)
    assert len(units) == 1
    assert units[0].text.strip().startswith("function add")
    assert units[0].text.strip().endswith("}")
