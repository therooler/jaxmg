# setup.py
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        # Tell wheel/setuptools: this project contains native (non-pure) code
        return True

    def is_pure(self):
        # Explicitly say it's not a pure Python package
        return False


setup(distclass=BinaryDistribution)
