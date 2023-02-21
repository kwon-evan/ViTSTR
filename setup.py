from setuptools import setup, find_packages

setup(
    name="ViTSTR",
    version="0.0.4",
    author="Heonjin Kwon",
    author_email="kwon@4ind.co.kr",
    description="ViTSTR written in pytorch-lightning",
    keywords=["pytorch", "pytorch-lightning", "license-plate-recognition", "OCR"],
    install_requires=[
        "pytorch-lightning>=1.7.0, <=1.9.0",
        "numpy>=1.17.1",
        "Pillow==9.4.0",
        "nltk==3.8.1",
        # 'imutils>=0.4.0',
        "timm >= 0.4.12",
        "multidict",
        "attrs",
        "yarl",
        "idna_ssl",
        "charset_normalizer",
        "aiosignal",
    ],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
