import setuptools
from distutils.core import setup

setup(
    name='solarvino',
    version='0.0.1',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy==1.19.5',
        'opencv-python==4.5.3.56',
        'openvino==2021.4.0'
    ]
)
