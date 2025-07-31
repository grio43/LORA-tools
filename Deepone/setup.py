from pathlib import Path

from setuptools import find_packages, setup

# -------------------------------------------------
# Core metadata — every literal must be *quoted*
# -------------------------------------------------
PACKAGE_NAME = "camie_distill"
VERSION = "0.1.0"        # <- quotes added; 0.1.0 without quotes is illegal Python
DESCRIPTION = "Student‑teacher distillation toolkit for Camie‑Tagger"

THIS_DIR = Path(__file__).parent
LONG_DESCRIPTION = (THIS_DIR / "README.md").read_text(encoding="utf-8")

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "examples")),
    install_requires=[
        "torch>=2.1",
        "timm==0.9.12",
        "safetensors>=0.5.3",
        # Use the pre‑built wheel on Windows; comment out if you *must* build from source
        "deepspeed>=0.16.3,<0.17",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    zip_safe=False,
)