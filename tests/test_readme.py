from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
README_ZH_PATH = ROOT / "README_zh.md"
README_ZH = README_ZH_PATH.read_text(encoding="utf-8") if README_ZH_PATH.exists() else ""


def test_readmes_switch_language_above_project_links() -> None:
    assert "[简体中文](./README_zh.md)" in README
    assert "[English](./README.md)" in README_ZH
    assert README.index("[简体中文](./README_zh.md)") < README.index("[Code]")
    assert README_ZH.index("[English](./README.md)") < README_ZH.index("[代码]")


def test_english_and_chinese_content_are_separate() -> None:
    assert "## 中文说明" not in README
    assert "OpenAster-1 是一个完全开源" not in README
    assert "## 项目亮点" in README_ZH
    assert "## 模型列表" in README_ZH
    assert "## 推理" in README_ZH


def test_readme_documents_text_and_vision_gui_commands() -> None:
    assert "python inference/app.py --model binichallein/OpenAster1-4k-base" in README
    assert "python inference/app.py --model binichallein/OpenAster1-math" in README
    assert "python inference/app.py --model binichallein/OpenAster1-128k-base" in README
    assert "python inference/app.py --model binichallein/OpenAster1-VL" in README


def test_readme_documents_only_gui_inference() -> None:
    assert "inference/inference.py" not in README
    assert "assets/terminal-demo.gif" not in README
    assert "Terminal chat" not in README
    assert "assets/gui-demo.gif" not in README


def test_readme_references_three_gui_demo_gifs() -> None:
    demos = [
        "gui-vision-demo.gif",
        "gui-text-thinking-demo.gif",
        "gui-math-demo.gif",
    ]

    for demo in demos:
        assert f"assets/{demo}" in README
        assert (ROOT / "assets" / demo).is_file()


def test_release_has_exactly_one_inference_source_file() -> None:
    scripts = sorted(path.name for path in (ROOT / "inference").glob("*.py"))
    assert scripts == ["app.py"]


def test_terminal_demo_asset_is_removed() -> None:
    assert not (ROOT / "assets" / "terminal-demo.gif").exists()
