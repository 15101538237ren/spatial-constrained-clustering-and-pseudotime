from setuptools import setup, find_packages

setup(
    name='SpaceFlow',
    version='0.1.0',
    packages=find_packages(exclude=['tests*']),
    description='Identifying Spatiotemporal Patterns of Cells for Spatial Transcriptome Data.',
    author='Honglei Ren',
    author_email='hongleir1@gmail.com'
)