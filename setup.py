import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="som-summarizer",
    version="1.0.9",
    description="Summarize long ducuments of all types",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/abhinav-TB/text-summarization",
    author="Abhinav T B",
    author_email="abhinavtb@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    packages=["som_summarizer"],
    include_package_data=True,
    install_requires=["nltk==3.7", "sklearn-som==1.1.0","sentence-transformers==2.2.0","primefac==2.0.12"],
)
