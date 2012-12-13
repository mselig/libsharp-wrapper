/*
 *  This file is part of libsharp-wrapper.
 *
 *  libsharp-wrapper is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  libsharp-wrapper is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libsharp-wrapper; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */

/*
 *  libsharp-wrapper is being developed at the
 *  Max-Planck-Institute for Astrophysics.
 *
 *  Copyright (C) 2012 Max-Planck-Society
 *  Author: Marco Selig
 */

typedef struct
{   double re,im;
}
c128;

typedef struct
{   float re,im;
}
c64;

void a2m_d(const c128*,int,int,double*,int,int);
void m2a_d(const double*,int,int,c128*,int,int,bool);

void a2m_f(const c64*,int,int,float*,int,int);
void m2a_f(const float*,int,int,c64*,int,int,bool);

