from pathlib import Path
from typing import *

import setuptools

PACKAGE = "prata"
CURRENT_DIR = Path(__file__).resolve().parent

with open(CURRENT_DIR / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def read_requirements(path: Union[str, Path]):
    with open(path, "r") as fh:
        return {line.strip() for line in fh.readlines() if not line.startswith("#")}


__VERSION__ = "0.0.1"

requirements = list(read_requirements(CURRENT_DIR / "requirements.txt"))
extras_require = {"all": set()}


packages = setuptools.find_packages()
entry_points = {"console_scripts": (f"{PACKAGE} = {PACKAGE}.__main__:main",)}

setuptools.setup(
    name=PACKAGE,
    packages=packages,
    package_dir={"": str("src")},
    version=__VERSION__,
    author="trungdt",
    author_email="termanteus@gmail.com",
    description=PACKAGE,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    entry_points=entry_points,
    install_requires=requirements,
    extras_require=extras_require,
)
