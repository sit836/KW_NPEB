import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name='kwnpeb',
    version='0.1',
    scripts=['kwnpeb'],
    author="Sile Tao, Li Zhang",
    author_email="sile@ualberta.ca, lzhang2@ualberta.ca",
    description="Compute the Kiefer-Wolfowitz nonparametric maximum likelihood estimator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sit836/KW_NPEB",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
