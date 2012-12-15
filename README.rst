libsharp-wrapper
================

Python wrapper for `**libsharp** <http://sourceforge.net/projects/libsharp/>`_

Summary
-------

Description
...........

The **libsharp-wrapper** is a Python wrapper for the C-library
`**libsharp** <http://sourceforge.net/projects/libsharp/>`_. It is written
in Cython and C++. The functional layout is inspired by
`healpy <https://github.com/healpy/healpy>`_.

The wrapper provides access to the spherical harmonics transformation of the
Gauss-Legendre pixelization. An extension to all pixelizations supported by
**libsharp** would be feasible.

Features
........

- Spherical harmonics transformations from and to the Gauss-Legendre
  pixelization::

	map2alm              alm2map

- Support for NumPy arrays in double and float precision (function operating on
  floats are suffixed by ``_f``)
- Support functions for geometrical quantities

Installation
------------

Requirements
............

... for **libsharp**

- GNU make
- GNU gcc (v4.x)
- GNU autoconf
- git

... for **libsharp-wrapper**

- Cython (v0.1x)
- Python (v2.7.x)
- Python Development Tools
- Numpy
- Numpy Development Tools

Compilation
...........

... of **libsharp**

::

	git clone git://git.code.sf.net/p/libsharp/code libsharp-code
	cd libsharp-code
	autoconf
	configure --enable-pic --disable-openmp
	make
	cd ..

... of **libsharp-wrapper**

::

	git clone git://github.com/mselig/libsharp-wrapper.git libsharp-wrapper
	cd libsharp-wrapper
	python setup.py build_ext

Installation
............

::

	python setup.py install

Alternatively, a private or user specific installation can be done by::

	python setup.py install --user
	python setup.py install --install-lib=/SOMEWHERE

Test
....

In python run::

	>>> import numpy as np
	>>> import libsharp_wrapper_gl as gl
	>>> gl.map2alm(np.ones(28))

Release Notes
-------------

The **libsharp-wrapper** is licensed under the
`GPLv2 <http://www.gnu.org/licenses/old-licenses/gpl-2.0.html>`_
and is distributed *without any warranty*.

The current version is tagged **v0.1**.

