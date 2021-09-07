from setuptools import Extension, setup

module = Extension("spkmeans", sources=['spkmeans.c', 'spkmeansmodule.c'])
setup(
    name='spkmeans',
    version='1.0',
    author='Ben_&_Ofek',
    description='Spectral clustering module',
    ext_modules=[module]
)
