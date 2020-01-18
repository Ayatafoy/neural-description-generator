from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Neural description generator',
    version='0.1',
    description='The model generates product description based on image and other attributes',
    classifiers=['Programming Language :: Python :: 3.7'],
    url='https://github.com/Ayatafoy/neural-description-generator.git',
    author='Aleksey Romanov, Artur Nizamutdinov',
    author_email='aromanov@griddynamics.com, nizamutdin.art@gmail.com',
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False
)
