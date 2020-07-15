from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')


setup(
    name='objectdetection',
    version='0.1',
    description='Object detection code for grasping',
    url='https://github.com/caiobarrosv/object_detection_for_grasping',
    author='Caio Viturino, Daniel Oliveira, Kleber Santana',
    author_email='engcaiobarros@gmail.com, dandmetal@gmail.com, engkleberf@gmail.com',
    package_dir={'': 'train_utils'},
    packages=find_packages(where='train_utils'),
    install_requires=['numpy'],
    long_description=long_description,
)