import pathlib
from setuptools import setup, find_packages

ROOT_DIR = pathlib.Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text()
VERSION = (ROOT_DIR / "VERSION").read_text()

setup(
    name                = "UtilsRL", 
    version             = VERSION, 
    description         = "A python module desgined for RL logging, monitoring and experiments managing. ", 
    long_description    = README,
    long_description_content_type = "text/markdown",
    url                 = "https://github.com/typoverflow/UtilsRL",
    author              = "typoverflow", 
    author_email        = "typoverflow@outlook.com", 
    license             = "MIT", 
    packages            = ["UtilsRL"],
    include_package_data = True, 
    install_requires = [
        "numpy>=1.18.0", 
        "cloudpickle>=1.2.0", 
    ], 
    tests_require=["pytest", "mock"], 
    python_requires=">=3.7", 
)