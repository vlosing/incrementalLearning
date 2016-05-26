 
from distutils.core import setup, Extension

module1 = Extension('libpythoninterface',
                    include_dirs = ['/hri/sit/latest/External/AllPython/2.7/lucid64/include/python2.7',
                    '/home/vlosing/C2-Update-Jeffrey/eclipse_workspace_pub/grlvq_c/src/',
                    '/hri/sit/latest/External/AllPython/2.7/lucid64/lib/numpy/core/include/numpy/'],
                    libraries = ['grlvqc','pthread'],
                    library_dirs = ['/home/vlosing/C2-Update-Jeffrey/eclipse_workspace_pub/grlvq_c/Debug'],
                    sources = ['intf.c'])

setup (name = 'PackageName',
       version = '1.0',
       description = 'This is a demo package',
       author = 'Martin v. Loewis',
       author_email = 'martin@v.loewis.de',
       url = 'http://docs.python.org/extending/building',
       long_description = '''
This is really just a demo package.
''',
       ext_modules = [module1])