from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="taxonova",
    version="25.11.0",
    author="Kardokh Kaka Bra",
    author_email="kardokhkaka@gmail.com",  
    description="Deep Learning-based Taxonomic Classification for Amplicon Sequences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kardokhk/taxonova",  
    project_urls={
        "Bug Tracker": "https://github.com/kardokhk/taxonova/issues",
        "Documentation": "https://github.com/kardokhk/taxonova#readme",
        "Source Code": "https://github.com/kardokhk/taxonova",
    },
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'taxonova=taxonova.main:main',
            'taxonova-setup=taxonova.utils.model_downloader:post_install_setup',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    keywords=[
        "bioinformatics",
        "taxonomy",
        "classification",
        "deep-learning",
        "amplicon",
        "16S",
        "rRNA",
        "microbiome",
        "metagenomics",
    ],
    zip_safe=False,
)
