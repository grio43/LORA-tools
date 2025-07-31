# Deepone/setup.py
from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------------
# Robust long‑description loader — will not crash if README is absent
# ------------------------------------------------------------------
def read_long_description() -> str:
    """
    Search for a README.md in the current folder or one directory
    above; return an empty string if nothing is found.
    """
    for candidate in (ROOT / "README.md", ROOT.parent / "README.md"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return ""  # fallback keeps 'pip install' alive

setup(
    name="camie_distill",
    version="0.1.0",
    description="Student‑teacher distillation toolkit for Camie‑Tagger",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=[
        "torch>=2.1",
        "timm==0.9.12",
        "safetensors>=0.5.3",
        "deepspeed>=0.16.3,<0.17",
    ],
    include_package_data=True,
    zip_safe=False,
)