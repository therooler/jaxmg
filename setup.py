from setuptools import setup, find_packages

setup(
    name="jaxmg",
    version="0.1.0",
    author="Roeland Wiersema",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "python": ["libpotrf.so"],  # ship the shared library
    },
    install_requires=[
        "jax>=0.6.1",
    ],
    python_requires=">=3.9",
)