from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RecAll", 
    version="0.0.1",
    packages=find_packages(),
    install_requires=['torch>=1.8.0'],
    author="Francesco Imbriglia, Alessio Sciacchitano",
    license='MIT',
    classifiers=['Development Status :: 3 - Alpha',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Artificial Intelligence']
)