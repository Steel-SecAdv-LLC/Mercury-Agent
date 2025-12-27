"""
Setup configuration for OMNI ♱ AVA
"""

from pathlib import Path

from setuptools import find_packages, setup

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="omni-ava",
    version="1.0.0",
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
    python_requires=">=3.11",
    # Core dependencies only - lightweight for basic functionality
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "click>=8.1.0",
        "pydantic>=2.0.0",
        "bcrypt>=4.0.1",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "networkx>=3.0",
        "cryptography>=41.0.0",
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "httpx>=0.25.0",
    ],
    extras_require={
        # ML dependencies - heavy ML stack (torch, deepface, opencv)
        "ml": [
            "torch>=2.2.0",
            "pytorch-lightning>=2.0.0",
            "deepface>=0.0.79",
            "opencv-python>=4.8.0",
            "pillow>=10.0.0",
        ],
        # Development dependencies - testing and linting tools
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
        # Quantum simulation dependencies
        "quantum": [
            "qutip>=4.7.3",
        ],
        # Mathematical computation dependencies
        "math": [
            "sympy>=1.12",
            "mpmath>=1.3.0",
        ],
        # GUI/visualization dependencies
        "gui": [
            "streamlit>=1.26.0",
            "plotly>=5.16.0",
        ],
        # Documentation dependencies
        "docs": [
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
        ],
        # Full installation - all optional dependencies
        "full": [
            "torch>=2.2.0",
            "pytorch-lightning>=2.0.0",
            "deepface>=0.0.79",
            "opencv-python>=4.8.0",
            "pillow>=10.0.0",
            "qutip>=4.7.3",
            "streamlit>=1.26.0",
            "plotly>=5.16.0",
            "sympy>=1.12",
            "mpmath>=1.3.0",
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
