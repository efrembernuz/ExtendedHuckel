from setuptools import setup, Extension, find_packages


def get_version_number():
    main_ns = {}
    for line in open('huckelpy/__init__.py', 'r').readlines():
        if not(line.find('__version__')):
            exec(line, main_ns)
            return main_ns['__version__']


setup(name='huckelpy',
      version=get_version_number(),
      description='Package to calculate the molecular orbitals with the extended huckel method',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      author='Efrem Bernuz',
      author_email='komuisan@gmail.com',
      packages=find_packages(where="."),
      package_data={'': ['basis_set.yaml']},
      include_package_data=True,
      install_requires=['numpy', 'PyYAML', 'scipy'])
