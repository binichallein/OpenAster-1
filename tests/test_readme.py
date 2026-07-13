from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_readme_documents_one_shot_and_interactive_text_inference() -> None:
    assert "python inference/inference.py --model binichallein/OpenAster1-math" in README
    assert '--prompt "Solve: 24 * 17"' in README


def test_readme_documents_128k_context_inference() -> None:
    assert "--model binichallein/OpenAster1-128k-base" in README
    assert "--context-tokens 131072" in README


def test_readme_documents_one_shot_vision_inference() -> None:
    assert "--model binichallein/OpenAster1-VL" in README
    assert "--image /path/to/image.jpg" in README


def test_readme_documents_text_and_vision_gui_commands() -> None:
    assert "python inference/app.py --model binichallein/OpenAster1-math" in README
    assert "python inference/app.py --model binichallein/OpenAster1-VL" in README


def test_readme_references_both_animated_demos() -> None:
    assert "assets/terminal-demo.gif" in README
    assert "assets/gui-demo.gif" in README


def test_release_has_exactly_two_public_inference_scripts() -> None:
    scripts = sorted(path.name for path in (ROOT / "inference").glob("*.py"))
    assert scripts == ["app.py", "inference.py"]
