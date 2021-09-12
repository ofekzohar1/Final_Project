from setuptools import Extension, setup

module = Extension("spkmeansmodule", sources=['spkmeans.c', 'spkmeansmodule.c'])
setup(
    name='spkmeansmodule',
    version='1.1',
    author='Ben_&_Ofek',
    description='Spectral Clustering module',
    ext_modules=[module]
)
