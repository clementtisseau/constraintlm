from setuptools import setup, find_packages

setup(
    name="constraintlm",          # the name you’ll import with
    version="0.1.0",              # your package version
    packages=find_packages(),     # automatically find all packages
    install_requires=[
        "torch>=1.0",             # list your runtime dependencies here
    ],
)