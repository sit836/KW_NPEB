import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def get_requirements(path):
    requirements = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            requirements.append(line)

    return requirements


setuptools.setup(
    name='kwnpeb',
    version='0.1.6',
    author="Sile Tao, Li Zhang",
    author_email="sile@ualberta.ca, lzhang2@ualberta.ca",
    description="Compute the Kiefer-Wolfowitz nonparametric maximum likelihood estimator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sit836/KW_NPEB",
    packages=setuptools.find_packages(),
    install_requires=get_requirements('requirements.txt'),
    dependency_links=['https://download.mosek.com/stable/wheel/index.html'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
