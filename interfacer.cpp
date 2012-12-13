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

#include "sharp_cxx.h"
#include "interfacer.h"

void a2m_d(const c128* alm,int lmax,int mmax,double* map,int nlat,int nlon)
{   sharp_cxxjob<double> job;
    job.set_Gauss_geometry(nlat,nlon);
    job.set_triangular_alm_info(lmax, mmax);
    job.alm2map((double*)alm,map,false);
}

void m2a_d(const double* map,int nlat,int nlon,c128* alm,int lmax,int mmax,bool add_alm)
{   sharp_cxxjob<double> job;
    job.set_Gauss_geometry(nlat,nlon);
    job.set_triangular_alm_info(lmax,mmax);
    job.map2alm(map,(double*)alm,add_alm);
}

void a2m_f(const c64* alm,int lmax,int mmax,float* map,int nlat,int nlon)
{   sharp_cxxjob<float> job;
    job.set_Gauss_geometry(nlat,nlon);
    job.set_triangular_alm_info(lmax, mmax);
    job.alm2map((float*)alm,map,false);
}

void m2a_f(const float* map,int nlat,int nlon,c64* alm,int lmax,int mmax,bool add_alm)
{   sharp_cxxjob<float> job;
    job.set_Gauss_geometry(nlat,nlon);
    job.set_triangular_alm_info(lmax,mmax);
    job.map2alm(map,(float*)alm,add_alm);
}

