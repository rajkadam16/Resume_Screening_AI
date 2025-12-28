"""
Resume Screening AI - Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="resume-screening-ai",
    version="1.0.0",
    author="Raj Kadam",
    author_email="your.email@example.com",
    description="An intelligent resume screening system powered by ML and NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/resume-screening-ai",
    packages=find_packages(exclude=["tests", "scripts", "tools"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "resume-screener=app.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "app": ["templates/*.html"],
    },
)
