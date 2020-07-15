from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='object_detection_for_grasping',
    url='https://github.com/caiobarrosv/object_detection_for_grasping',
    author='Caio Viturino, Daniel Oliveira, Kleber Santana',
    author_email='engcaiobarros@gmail.com, dandmetal@gmail.com, engkleberf@gmail.com',
    # Needed to actually package something
    packages=['object_detection_for_grasping'],
    # Most dependencies are already installed in Google Colab
    install_requires=['numpy'],
    version='0.1',
    license='MIT',
    description='Object detection code for grasping',
    long_description=open('README.md').read(),
)