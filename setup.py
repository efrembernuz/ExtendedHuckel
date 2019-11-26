from numpy.distutils.core import setup, Extension


def get_version_number():
    for l in open('huckelpy/__init__.py', 'r').readlines():
        if not(l.find('__version__')):
            exec(l, globals())
            return __version__


setup(name='huckelpy',
      version=get_version_number(),
      description='Package to calculate the molecular orbitals with the extended huckel method',
      author='Efrem Bernuz',
      author_email='komuisan@gmail.com',
      packages=['huckelpy'],
      package_data={'': ['basis_set.yaml']},
      include_package_data=True,
      install_requires=['numpy', 'PyYAML', 'scipy'])