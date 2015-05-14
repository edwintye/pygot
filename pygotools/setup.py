from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

cmdclass = {}
ext_modules =[]
requires = [
            'cvxopt',
            'scipy>=0.10.0',
            'numpy>=1.6.0'
            ]

tests_require = [
                 'nose>=1.1'
                 ]

setup(name='pygotools',
      version='0.1.0',
      description='Global Optimization tools',
      long_description=[],#readme(),
      author="Edwin Tye",
      author_email="Edwin.Tye@gmail.com",
      packages=[
                'pygotools',
                'pygotools.direct',
                'pygotools.gradient',
                'pygotools.convex',
                'pygotools.optutils'
                ],
      license='LICENCE.txt',
      install_requires=requires,
      cmdclass = cmdclass,
      ext_modules = ext_modules,
      tests_require=tests_require,
      test_suite='nose.collector',
      scripts=[]
      )
