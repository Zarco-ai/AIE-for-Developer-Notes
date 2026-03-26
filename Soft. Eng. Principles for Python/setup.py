'''This file's purpose is to:
- specify the specific requirements(like libraries) and information on this package before installation.
- This is for developers; it list the minimum libraries needed to make your code work.
- It is used when you want to distribute your package so others can install it.

'''

from setuptools import setup

setup(name='my_package',
      version='0.0.1',
      description='An example package for DataCamp.',
      author='Christopher Zarco',
      author_email='zarco7452@icloud.com',
      packages=['characters'],
      install_requires=['matplotlib',
                        'numpy==1.15.4',
                        'pycodestyle>=2.4.0']) #Random libraries required lmao
