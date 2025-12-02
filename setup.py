"""
Setup configuration for OMNI ♱ AVA
"""

from setuptools import setup, find_packages
from pathlib import Path

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="omni-ava",
    version="0.1.0",
    description=(
        "Neural-symbolic AI archetype integrating ML-based anomaly detection, "
        "quantum simulations, and biometric analysis with ethical alignment protocols"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Steel Security Advisory LLC",
    author_email="support@steelsecurityadvisors.com",
    url="https://github.com/Steel-SecAdv-LLC/OMNI-AVA",
    license="GPL-3.0-or-later",
    packages=find_packages(where="src", exclude=["tests", "tests.*", "examples", "docs"]),
    package_dir={"": "src"},
    python_requires=">=3.12",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "torch>=2.0.0",
        "pytorch-lightning>=2.0.0",
        "pandas>=2.0.0",
        "click>=8.1.0",
        "pydantic>=2.0.0",
        "bcrypt>=4.0.1",
        "deepface>=0.0.79",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.1",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "bandit>=1.7.5",
            "safety>=2.3.5",
            "fastapi>=0.104.0",
        ],
        "quantum": [
            "qutip>=4.7.3",
        ],
        "gui": [
            "streamlit>=1.26.0",
            "plotly>=5.16.0",
        ],
        "docs": [
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "omni-ava=omni_anomaly_engine.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Quality Assurance",
    ],
    keywords=(
        "anomaly-detection machine-learning biometrics quantum "
        "astrophysics security neural-networks"
    ),
    project_urls={
        "Bug Reports": ("https://github.com/Steel-SecAdv-LLC/OMNI-AVA/issues"),
        "Source": "https://github.com/Steel-SecAdv-LLC/OMNI-AVA",
        "Documentation": "https://omni-ava.readthedocs.io/",
    },
)
