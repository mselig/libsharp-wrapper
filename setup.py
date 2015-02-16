## This file is part of libsharp-wrapper.
##
## libsharp-wrapper is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## libsharp-wrapper is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with libsharp-wrapper; if not, write to the Free Software
## Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

## libsharp-wrapper is being developed at the
## Max-Planck-Institute for Astrophysics.
##
## Copyright (C) 2012 Max-Planck-Society
## Author: Marco Selig

from distutils.core import setup
from distutils.extension import Extension as extension
from Cython.Distutils import build_ext
from numpy import get_include as numpy_get_include

srcs = ["wrapper.pyx",
        "interfacer.cpp"]

libs = ["sharp",
        "fftpack",
        'c_utils']

idirs = [numpy_get_include(),
         "../libsharp-code/c_utils",
         "../libsharp-code/libfftpack",
         "../libsharp-code/libsharp"]

ldirs = ["../libsharp-code/auto/lib"]

exmod = [extension("libsharp_wrapper_gl",
                   srcs,
                   language="c++",
                   include_dirs=idirs,
                   library_dirs=ldirs,
                   libraries=libs)]

setup(name="libsharp_wrapper_gl",
      version="0.2",
      author="Marco Selig",
      author_email="mselig@mpa-garching.mpg.de",
      cmdclass={"build_ext": build_ext},
      ext_modules=exmod)