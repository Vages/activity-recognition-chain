# Learnt from: http://python-packaging.readthedocs.io/en/latest/dependencies.html

from setuptools import setup

setup(name='activity-recognition-chain',
      version='0.1.3',
      description='An implementation of the Activity Recognition Chain described in Bulling et al.’s 2014 paper “A '
                  'Tutorial on Human Activity Recognition Using Body-Worn Inertial Sensors” with Scikit-learn’s '
                  'Random Forests Classifiers',
      url='https://github.com/Vages/activity-recognition-chain',
      author='Eirik Vågeskar',
      author_email='eirikvageskar(at)gmail(dot)com',
      packages=['acrechain'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'pandas',
      ],
      include_package_data=True,
      zip_safe=False)
