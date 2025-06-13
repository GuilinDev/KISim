#!/usr/bin/env python3
"""
Setup script for KISim
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="kisim",
    version="1.0.0",
    author="KISim Contributors",
    author_email="your.email@university.edu",
    description="Kubernetes Intelligent Scheduling Simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/KISim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
        ],
        "gpu": [
            "torch>=1.12.0+cu118",
            "torchvision>=0.13.0+cu118",
            "torchaudio>=0.12.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "kisim-train=scripts.rl.train_rl_agent:main",
            "kisim-evaluate=scripts.rl.evaluate_rl_agent:main",
            "kisim-plots=analysis.generate_plots:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.md", "*.txt"],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/KISim/issues",
        "Source": "https://github.com/yourusername/KISim",
        "Documentation": "https://github.com/yourusername/KISim/blob/main/docs/",
    },
)
