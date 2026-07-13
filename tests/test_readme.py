from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_documents_text_and_vision_gui_commands() -> None:
    assert "python inference/app.py --model binichallein/OpenAster1-math" in README
    assert "python inference/app.py --model binichallein/OpenAster1-128k-base" in README
    assert "python inference/app.py --model binichallein/OpenAster1-VL" in README


def test_readme_documents_only_gui_inference() -> None:
    assert "inference/inference.py" not in README
    assert "assets/terminal-demo.gif" not in README
    assert "Terminal chat" not in README
    assert "assets/gui-demo.gif" in README


def test_release_has_exactly_one_inference_source_file() -> None:
    scripts = sorted(path.name for path in (ROOT / "inference").glob("*.py"))
    assert scripts == ["app.py"]


def test_terminal_demo_asset_is_removed() -> None:
    assert not (ROOT / "assets" / "terminal-demo.gif").exists()
