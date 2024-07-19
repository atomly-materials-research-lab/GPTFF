# from __future__ import annotations
import numpy as np

from setuptools import Extension, setup, find_packages

setup(
    # packages=find_packages(include=['gptff*'], exclude=['test*']),
    ext_modules=[
        Extension(
            name="gptff.utils_.compute_tp", 
            sources=["gptff/utils_/compute_tp.pyx"], 
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++11"],
        ),
        Extension(
            name="gptff.utils_.compute_nb", 
            sources=["gptff/utils_/compute_nb.pyx"], 
            include_dirs=[np.get_include()],
            language="c++",
            extra_compile_args=["-std=c++11"],
        ),
    ],
    setup_requires=["Cython"]
)