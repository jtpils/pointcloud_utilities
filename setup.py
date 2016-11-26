from setuptools import setup, find_packages

setup(name='pointcloud_utilities',
      version='0.1',
      description='Visualise and manipulate point cloud data',
      classifiers=['Development Status :: 2 - Pre-Alpha',
                   'Programming Language :: Python'],
      keywords='pointcloud lidar ALS TLS',
      url='https://github.com/stainbank/pointcloud_utilities',
      packages=find_packages(),
      install_requires=['laspy', 'matplotlib', 'numpy'])
