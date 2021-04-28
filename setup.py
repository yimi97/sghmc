from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='sghmc-2021',
      version='0.3',
      description='Implementation of Stochastic Gradient Hamiltonian Monte Carlo',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/yimi97/sghmc',
      author='Qinzhe Wang & Yi Mi',
      author_email='sif1900@outlook.com',
      license='MIT',
      packages=['sghmc'],
      zip_safe=False)