from setuptools import setup, find_packages
import os


def _get_requirements(file_name="requirements.txt"):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, file_name)) as f:
        requirements = f.read().splitlines()
        if not requirements:
            raise RuntimeError(f"Unable to read requirements from the {file_name} file.")
    return requirements


setup(name='BBOMol',
      version='0.2.0',
      description='Surrogate-based black-box optimization method for molecular properties',
      url='https://github.com/jules-leguy/BBOMol',
      author='Jules Leguy',
      author_email='leguy.jules@gmail.com',
      install_requires=_get_requirements(),
      license='LGPL',
      packages=find_packages(),
      zip_safe=False)
