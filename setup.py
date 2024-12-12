from setuptools import setup, find_packages

setup(
    name="pyRBDA",
    version="0.1.1",
    author="PC",
    author_email="premc946@gmail.com",
    description="Rigid-body spatial math in Python",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/prem-chand/pyspatial",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        # List your package dependencies here
        "numpy",
    ],
)
