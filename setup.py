from setuptools import setup, find_packages

setup(name='pointcloud_utilities',
      version='0.1',
      description='Visualise and manipulate point cloud data',
      classifiers=['Development Status :: 2 - Pre-Alpha',
                   'Programming Language :: Python'],
      keywords='pointcloud lidar ALS TLS',
      packages=find_packages(),
      install_requires=['laspy', 'matplotlib', 'numpy'])
