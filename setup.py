from setuptools import find_packages, setup
import os

# Get the directory containing the setup script
current_directory = os.path.abspath(os.path.dirname(__file__))

# Read the long description from the README file
with open(os.path.join(current_directory, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements from the requirements.txt file
with open(os.path.join(current_directory, "requirements.txt"), "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="mrz_reader",
    version="0.0.1a1",  # Alpha version
    description="MRZ Passport Reader From Image",
    packages=find_packages(where="mrz_reader"),  # This will find packages within the 'src' directory
    package_dir={"": "mrz_reader"},  # This tells setuptools that packages are under the 'src' directory
    python_requires=">=3.10.12",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="serdarhelli",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
)
