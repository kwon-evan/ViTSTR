from setuptools import setup, find_packages

setup(
    name="ViTSTR",
    version="0.0.5",
    author="Heonjin Kwon",
    author_email="kwon@4ind.co.kr",
    description="ViTSTR written in pytorch-lightning",
    keywords=["pytorch", "pytorch-lightning", "license-plate-recognition", "OCR"],
    install_requires=[
        "dataclasses",
    ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.0",
)
