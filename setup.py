from setuptools import setup


setup(name='SimpleRL',
      author='lccasagrande',
      license="MIT",
      version='0.1',
      python_requires='>=3.7',
      extras_require={
          'tf': ['tensorflow>=2.0.0'],
          'tf_gpu': ['tensorflow-gpu>=2.0.0'],
      },
      install_requires=[
          'gym',
          'numpy',
          'pandas',
          'cloudpickle',
      ])
