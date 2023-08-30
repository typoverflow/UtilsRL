import os
import pathlib
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext

def get_version() -> str:
    init = open(os.path.join("UtilsRL", "__init__.py"), "r").read().split()
    return init[init.index("__version__") + 2][1:-1]

ROOT_DIR = pathlib.Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text()
VERSION = get_version()

def get_install_requires():
    return [
        "gym>=0.23.1,<=0.24.1", 
        "tqdm", 
        "numpy", 
        "torch", 
        "pandas", 
        "opencv-python"
    ]

def get_extras_require():
    return {}

ext_modules = [
    Pybind11Extension("data_structure", 
        ["UtilsRL/data_structure/data_structure.cc"], 
    )
]

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
    packages            = find_packages(),
    include_package_data = True, 
    tests_require=["pytest", "mock"], 
    python_requires=">=3.7", 
    install_requires = get_install_requires(), 
    # extras_require = get_extras_require(), 
    ext_modules = ext_modules, 
    cmdclass = {"build_ext": build_ext}, 
)