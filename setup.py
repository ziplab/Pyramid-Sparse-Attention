"""
Setup script for Pyramid Sparse Attention (PSA)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Pyramid Sparse Attention for efficient video generation inference"

setup(
    name="psa-triton",
    version="0.1.0",
    author="PSA Team",
    author_email="your.email@example.com",
    description="Training-free inference acceleration for video generation models using Pyramid Sparse Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/PyramidSparseAttention",
    package_dir={"": "src"},
    packages=find_packages(where="src", exclude=["*.test", "*.test.*", "test.*", "test"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.6.0",
        "triton>=3.5.0",
        "transformers==4.56.0",
        "diffusers==0.35.2",
        "einops>=0.6.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml"],
    },
    zip_safe=False,
)
