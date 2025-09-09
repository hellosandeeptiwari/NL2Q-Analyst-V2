"""
NL2Q Analyst - Natural Language to Query Agent
A sophisticated pharmaceutical data analysis platform with AI-powered query generation.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#') and not line.startswith('-r')]

setup(
    name="nl2q-analyst",
    version="2.0.0",
    author="Sandeep Tiwari",
    author_email="sandeep@example.com",
    description="AI-powered pharmaceutical data analysis platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hellosandeeptiwari/NL2Q-Analyst",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Database :: Front-Ends",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "docker": read_requirements("requirements-docker.txt"),
    },
    entry_points={
        "console_scripts": [
            "nl2q-server=backend.main:app",
            "nl2q-setup=scripts.setup:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml", "*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/hellosandeeptiwari/NL2Q-Analyst/issues",
        "Source": "https://github.com/hellosandeeptiwari/NL2Q-Analyst",
        "Documentation": "https://github.com/hellosandeeptiwari/NL2Q-Analyst/wiki",
    },
)
