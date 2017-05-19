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

"""
           __     __   __                  __
         /  /   /__/ /  /                /  /
        /  /    __  /  /___    _______  /  /___    ____ __   _____   ______
       /  /   /  / /   _   | /  _____/ /   _   | /   _   / /   __/ /   _   |
      /  /_  /  / /  /_/  / /_____  / /  / /  / /  /_/  / /  /    /  /_/  /
      \___/ /__/  \______/ /_______/ /__/ /__/  \______| /__/    /   ____/
                                                                /__/
      __     __   _____   ____ __   ______    ______    _______   _____
     |  |/\/  / /   __/ /   _   / /   _   | /   _   | /   __  / /   __/
     |       / /  /    /  /_/  / /  /_/  / /  /_/  / /  /____/ /  /
     |__/\__/ /__/     \______| /   ____/ /   ____/  \______/ /__/     by Marco Selig (2012)
                               /__/      /__/

    libsharp_wrapper_gl.so
    > wrapper for libsharp restricted to Gauss-Legendre quadrature of the sphere

    implemented methods:
    - alm2map()          - lm2i()          - alm2map_f()
    - map2alm()          - i2lm()          - map2alm_f()
    - anaalm()           - pix2nn()        - anaalm_f()
    - anafast()                            - anafast_f()
    - synalm()                             - synalm_f()
    - synfast()                            - synfast_f()
    - dotlm()                              - dotlm_f()
    - smoothalm()
    - smoothmap()
    - vol()
    - weight()                             - weight_f()
    - bounds()
    - ang2pix()
    - pix2ang()
    - ang2xyz()
    - xyz2ang()
    - pix2xyz()
    - xyz2pix()

    global variable:
    - pi

"""
## compiler directives: (the next 2 lines are NO comments)
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as np


cdef extern from "math.h":
    double sqrt(double)
    double fabs(double)
    double ceil(double)
    double log(double)
    double exp(double)
    double cos(double)
    double sin(double)
    double acos(double)
    double atan2(double y,double x)

cdef extern from "interfacer.h":
    ctypedef struct c64:
        double re,im
    ctypedef struct c128:
        double re,im
    void a2m_f(c64* alm,int lmax,int mmax,float* map,int nlat,int nlon)
    void m2a_f(float* map,int nlat,int nlon,c64* alm,int lmax,int mmax,bint add_alm)
    void a2m_d(c128* alm,int lmax,int mmax,double* map,int nlat,int nlon)
    void m2a_d(double* map,int nlat,int nlon,c128* alm,int lmax,int mmax,bint add_alm)


cpdef double pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679


##-----------------------------------------------------------------------------

cdef void check_para4alen(int alen,int* p_lmax,int* p_mmax) except *:
    """
        check_para4alen(int alen,int* p_lmax,int* p_mmax)
        > checks whether parameters agree with the length of a spherical harmonics transform

    """
    if(alen<>(p_lmax[0]+1)*(p_mmax[0]+1)-(p_mmax[0]*(p_mmax[0]+1))/2):
        raise ValueError("len(a)=="+str(alen)+" mismatches (lmax,mmax)==("+str(p_lmax[0])+","+str(p_mmax[0])+")")

cdef void check_para4mlen(int mlen,int* p_nlat,int* p_nlon) except *:
    """
        check_para4mlen(int alen,int* p_nlat,int* p_nlon)
        > checks whether parameters agree with the length of a Gauss-Legendre map

    """
    if(mlen<>p_nlat[0]*p_nlon[0]):
        raise ValueError("len(m)=="+str(mlen)+" mismatches (nlat,nlon)==("+str(p_nlat[0])+","+str(p_nlon[0])+")")

##-----------------------------------------------------------------------------

cdef void set_para4alen(int alen,int* p_nlat,int* p_nlon,int* p_lmax,int* p_mmax) except *:
    """
        set_para4alen(int alen,int* p_nlat,int* p_nlon,int* p_lmax,int* p_mmax)
        > computes the parameters given the length of a spherical harmonics transform

    """
    if(p_lmax[0]<0):
        if(p_mmax[0]<0)or(p_mmax[0]>p_lmax[0]):
            p_lmax[0] = (-3+int(sqrt(1+8*alen)))/2
            p_mmax[0] = p_lmax[0]
        else:
            p_lmax[0] = (alen+(p_mmax[0]+1)*p_mmax[0]/2)/(p_mmax[0]+1)-1
    elif(p_mmax[0]<0)or(p_mmax[0]>p_lmax[0]):
            p_mmax[0] = p_lmax[0]
    check_para4alen(alen,p_lmax,p_mmax)
    if(p_nlat[0]<0):
        p_nlat[0] = p_lmax[0]+1
    if(p_nlon[0]<0):
        p_nlon[0] = 2*p_lmax[0]+1

cdef void set_para4mlen(int mlen,int* p_nlat,int* p_nlon,int* p_lmax,int* p_mmax) except *:
    """
        set_para4mlen(int mlen,int* p_nlat,int* p_nlon,int* p_lmax,int* p_mmax)
        > computes the parameters given the length of a Gauss-Legendre map

    """
    if(p_nlat[0]<0):
        if(p_nlon[0]<0):
            p_nlat[0] = int(sqrt(0.5*mlen))+1
            p_nlon[0] = 2*p_nlat[0]-1
        else:
            p_nlat[0] = mlen/p_nlon[0]
    elif(p_nlon[0]<0):
        p_nlon[0] = mlen/p_nlat[0]
    check_para4mlen(mlen,p_nlat,p_nlon)
    if(p_lmax[0]<0):
        p_lmax[0] = p_nlat[0]-1
    if(p_mmax[0]<0)or(p_mmax[0]>p_lmax[0]):
        p_mmax[0] = p_lmax[0]

##-----------------------------------------------------------------------------

cpdef int lm2i(np.ndarray[np.int_t,ndim=1,mode='c'] lm,int lmax):
    """
        lm2i(np.ndarray[np.int_t,ndim=1,mode='c'] lm,int lmax)
        > computates the index of a spherical harmonics component for a given [l,m] tuple

        input:
        - lm    vector containing [l,m]
        - lmax  maximum l of the transform

        output:
        - i     index

    """
    return <int> lm[0]+lm[1]*lmax-lm[1]*(lm[1]-1)/2

cpdef np.ndarray[np.int_t,ndim=1,mode='c'] i2lm(int i,int lmax):
    """
        i2lm(int lmax,int i)
        > computates the [l,m] tuple for a given index of a spherical harmonics component

        input:
        - i     index
        - lmax  maximum l of the transform

        output:
        - lm    vector containing [l,m]

    """
    cdef np.ndarray[np.int_t,ndim=1,mode='c'] lm = np.empty(2,dtype=np.int)
    lm[1] = int(ceil(((2*lmax+1)-sqrt((2*lmax+1)*(2*lmax+1)-8*(i-lmax)))/2)) ## int to np.int
    lm[0] = i-lm[1]*lmax+lm[1]*(lm[1]-1)/2
    return lm

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.int_t,ndim=1,mode='c'] pix2nn(int pix,int nlat,int nlon=-1):
    """
        pix2nn(int nlat,int nlon,int pix)
        > computates the all nearest neighbouring pixels sharing one edge with the give pixel

        input:
        - pix   pixel index
        - nlat  number of latitudinal bins or "rings"

        parameter:
        - nlon  number of longitudinal bins (defalut: 2*nlat-1)

        output:
        - nn    numpy array of nearest neighbouring pixel indices (note: len(n) varies)

    """
    if(nlon<0):
        nlon = 2*nlat-1
    cdef int lat = pix/nlon
    cdef np.ndarray[np.int_t,ndim=1,mode='c'] nn
    if(lat==0):
        nn = np.array([lat*nlon+(pix-1)%nlon,lat*nlon+(pix+1)%nlon,pix+nlon],dtype=np.int)
    elif(lat==nlat-1):
        nn = np.array([pix-nlon,lat*nlon+(pix-1)%nlon,lat*nlon+(pix+1)%nlon],dtype=np.int)
    else:
        nn = np.array([pix-nlon,lat*nlon+(pix-1)%nlon,lat*nlon+(pix+1)%nlon,pix+nlon],dtype=np.int)
    return nn

##-----------------------------------------------------------------------------



##-----------------------------------------------------------------------------

cpdef np.ndarray[np.float32_t,ndim=1,mode='c'] anaalm_f(np.ndarray[np.complex64_t,ndim=1,mode='c'] a,int lmax=-1,int mmax=-1):
    """
        anaalm_f(np.ndarray[np.complex64_t,ndim=1,mode='c'] a,int lmax=-1,int mmax=-1)
        > computates the angular power spectrum of a given spherical harmonics transform
        > float precision

        input:
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

        parameters:
        - lmax  maximum l of the transform (assumed default obtained from len(a))
        - mmax  maximum m of the transform (assumed default: lmax)

        output:
        - c     angular power spectrum (note: len(c)==lmax+1)

    """
    if(lmax<0):
        if(mmax<0)or(mmax>lmax):
            lmax = (-3+int(sqrt(1+8*len(a))))/2
            mmax = lmax
        else:
            lmax = (<int> len(a)+(mmax+1)*mmax/2)/(mmax+1)-1
    elif(mmax<0)or(mmax>lmax):
            mmax = lmax
    check_para4alen(<int> len(a),&lmax,&mmax)
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] c = np.zeros(lmax+1,dtype=np.float32)
    cdef float* p_c = <float*> c.data
    cdef c64* p_a = <c64*> a.data
    cdef int ll,mm,ii = 0
    for ll in range(0,lmax+1):
        p_c[ll] += (p_a[ii].re*p_a[ii].re)/(2*ll+1)
        ii += 1
    for mm in range(1,mmax+1):
        for ll in range(mm,lmax+1):
            p_c[ll] += 2*(p_a[ii].re*p_a[ii].re+p_a[ii].im*p_a[ii].im)/(2*ll+1)
            ii += 1
    return c

##-----------------------------------------------------------------------------

def alm2map_f(np.ndarray[np.complex64_t,ndim=1,mode='c'] a,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,bint cl=False):
    """
        alm2map_f(np.ndarray[np.complex64_t,ndim=1,mode='c'] a,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,bint cl=False)
        > computates the Gauss-Legendre map from a given spherical harmonics transform
        > float precision

        input:
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

        parameters:
        - nlat  number of latitudinal bins or "rings"               (default: lmax+1)
        - nlon  number of longitudinal bins                         (default: 2*lmax+1)
        - lmax  maximum l of the transform                          (assumed default obtained from len(a))
        - mmax  maximum m of the transform                          (assumed default: lmax)
        - cl    whether to return the angular power spectrum or not (default: False)

        output:
        - m     Gauss-Legendre map     (note: len(m)==nlat*nlon)
        - c     angular power spectrum (note: len(c)==lmax+1)

    """
    if(nlat<0)or(nlon<0)or(lmax<0)or(mmax<0):
        set_para4alen(<int> len(a),&nlat,&nlon,&lmax,&mmax)
    else:
        check_para4alen(<int> len(a),&lmax,&mmax)
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] m = np.empty(nlat*nlon,dtype=np.float32)
    a2m_f(<c64*> a.data,lmax,mmax,<float*> m.data,nlat,nlon)
    if(cl):
        return m,anaalm_f(a,lmax,mmax)
    else:
        return m

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.complex64_t,ndim=1,mode='c'] map2alm_f(np.ndarray[np.float32_t,ndim=1,mode='c'] m,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1):
    """
        map2alm_f(np.ndarray[np.float32_t,ndim=1,mode='c'] m,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1)
        > computates the spherical harmonics transform of a given Gauss-Legendre map
        > float precision

        input:
        - m     Gauss-Legendre map (note: len(m)==nlat*nlon)

        parameters:
        - nlat  number of latitudinal bins or "rings" (assumed default obtained from len(m))
        - nlon  number of longitudinal bins           (assumed default obtained from len(m))
        - lmax  maximum l of the transform            (default: nlat-1)
        - mmax  maximum m of the transform            (default: lmax)

        output:
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

    """
    if(nlat<0)or(nlon<0)or(lmax<0)or(mmax<0):
        set_para4mlen(<int> len(m),&nlat,&nlon,&lmax,&mmax)
    else:
        check_para4mlen(<int> len(m),&nlat,&nlon)
    cdef np.ndarray[np.complex64_t,ndim=1,mode='c'] a = np.empty((lmax+1)*(mmax+1)-(mmax*(mmax+1))/2,dtype=np.complex64)
    m2a_f(<float*> m.data,nlat,nlon,<c64*> a.data,lmax,mmax,False)
    return a

##-----------------------------------------------------------------------------

def anafast_f(np.ndarray[np.float32_t,ndim=1,mode='c'] m,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,bint alm=False):
    """
        anafast_f(np.ndarray[np.float32_t,ndim=1,mode='c'] m,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,bint alm=False)
        > computates the angular power spectrum of a given Gauss-Legendre map
        > float precision

        input:
        - m     Gauss-Legendre map (note: len(m)==nlat*nlon)

        parameters:
        - nlat  number of latitudinal bins or "rings"               (assumed default obtained from len(m))
        - nlon  number of longitudinal bins                         (assumed default obtained from len(m))
        - lmax  maximum l of the transform                          (default: nlat-1)
        - mmax  maximum m of the transform                          (default: lmax)
        - alm   whether to return the transform additionally or not (default: False)

        output:
        - c     angular power spectrum        (note: len(c)==lmax+1)
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

    """
    if(nlat<0)or(nlon<0)or(lmax<0)or(mmax<0):
        set_para4mlen(<int> len(m),&nlat,&nlon,&lmax,&mmax)
    else:
        check_para4mlen(<int> len(m),&nlat,&nlon)
    cdef np.ndarray[np.complex64_t,ndim=1,mode='c'] a = np.empty((lmax+1)*(mmax+1)-(mmax*(mmax+1))/2,dtype=np.complex64)
    m2a_f(<float*> m.data,nlat,nlon,<c64*> a.data,lmax,mmax,False)
    if(alm):
        return anaalm_f(a,lmax,mmax),a
    else:
        return anaalm_f(a,lmax,mmax)

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.complex64_t,ndim=1,mode='c'] synalm_f(np.ndarray[np.float32_t,ndim=1,mode='c'] c,int lmax=-1,int mmax=-1):
    """
        synalm(np.ndarray[np.float32_t,ndim=1,mode='c'] c,int lmax=-1,int mmax=-1)
        > synthesises a spherical harmonics transform from a given angular power spectrum
        > float precision

        input:
        - c     angular power spectrum (note: len(c)==lmax+1)

        parameters:
        - lmax  maximum l of the transform (assumed default: len(c)-1)
        - mmax  maximum m of the transform (assumed default: lmax)

        output:
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

    """
    if(lmax<0)or(lmax>=(<int> len(c))):
        lmax = <int> len(c)-1
    if(mmax<0)or(mmax>lmax):
        mmax = lmax
    cdef np.ndarray[np.complex64_t,ndim=1,mode='c'] a = np.empty((lmax+1)*(mmax+1)-(mmax*(mmax+1))/2,dtype=np.complex64)
    cdef c64* p_a = <c64*> a.data
    cdef float* p_c = <float*> c.data
    cdef int ll,mm,ii = 0
    for ll in range(0,lmax+1):
        p_a[ii].re = float(np.random.normal(0,np.sqrt(p_c[ll]))) ## alternative?
        p_a[ii].im = 0
        ii += 1
    for mm in range(1,mmax+1):
        for ll in range(mm,lmax+1):
            p_a[ii].re = float(np.random.normal(0,np.sqrt(0.5*p_c[ll]))) ## alternative?
            p_a[ii].im = float(np.random.normal(0,np.sqrt(0.5*p_c[ll]))) ## alternative?
            ii += 1
    return a

##-----------------------------------------------------------------------------

def synfast_f(np.ndarray[np.float32_t,ndim=1,mode='c'] c,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,alm=False):
    """
        synfast(np.ndarray[np.float32_t,ndim=1,mode='c'] c,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,alm=False)
        > synthesises a spherical harmonics transform from a given angular power spectrum
        > float precision

        input:
        - c     angular power spectrum (note: len(c)==lmax+1)

        parameters:
        - nlat  number of latitudinal bins or "rings"               (default: lmax+1)
        - nlon  number of longitudinal bins                         (default: 2*lmax+1)
        - lmax  maximum l of the transform                          (assumed default: len(c)-1)
        - mmax  maximum m of the transform                          (assumed default: lmax)
        - alm   whether to return the transform additionally or not (default: False)

        output:
        - m     Gauss-Legendre map            (note: len(m)==nlat*nlon)
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

    """
    if(lmax<0)or(lmax>=(<int> len(c))):
        lmax = <int> len(c)-1
    if(mmax<0)or(mmax>lmax):
        mmax = lmax
    cdef np.ndarray[np.complex64_t,ndim=1,mode='c'] a = synalm_f(c,lmax,mmax)
    if(nlat<0):
        nlat = lmax+1
    if(nlon<0):
        nlon = 2*lmax+1
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] m = alm2map_f(a,nlat,nlon,lmax,mmax,cl=False)
    if(alm):
        return m,a
    else:
        return m

##-----------------------------------------------------------------------------

cpdef float dotlm_f(np.ndarray[np.complex64_t,ndim=1,mode='c'] a, np.ndarray[np.complex64_t,ndim=1,mode='c'] b,int lmax=-1,int mmax=-1):
    """
        dotlm(np.ndarray[np.complex64_t,ndim=1,mode='c'] a, np.ndarray[np.complex64_t,ndim=1,mode='c'] b,int lmax=-1,int mmax=-1)
        > computates the inner product of two given spherical harmonics transform
        > float precision

        input:
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)
        - b     spherical harmonics transform (note: len(b)==len(a))

        parameters:
        - lmax  maximum l of the transform (assumed default obtained from len(a))
        - mmax  maximum m of the transform (assumed default: lmax)

        output:
        - dot   inner product

    """
    if(lmax<0):
        if(mmax<0)or(mmax>lmax):
            lmax = (-3+int(sqrt(1+8*len(a))))/2
            mmax = lmax
        else:
            lmax = (<int> len(a)+(mmax+1)*mmax/2)/(mmax+1)-1
    elif(mmax<0)or(mmax>lmax):
            mmax = lmax
    check_para4alen(<int> len(a),&lmax,&mmax)
    check_para4alen(<int> len(b),&lmax,&mmax)
    cdef c64* p_a = <c64*> a.data
    cdef c64* p_b = <c64*> b.data
    cdef float dot = 0.0
    cdef int ll,mm,ii = 0
    for ll in range(0,lmax+1):
        dot += p_a[ii].re*p_b[ii].re
        ii += 1
    for mm in range(1,mmax+1):
        for ll in range(mm,lmax+1):
            dot += 2*(p_a[ii].re*p_b[ii].re+p_a[ii].im*p_b[ii].im)
            ii += 1
    return dot

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.float32_t,ndim=1,mode='c'] weight_f(np.ndarray[np.float32_t,ndim=1,mode='c'] m,np.ndarray[np.float32_t,ndim=1,mode='c'] v,float p=1.0,int nlat=-1,int nlon=-1,bint overwrite=False):
    """
        weight(np.ndarray[np.float32_t,ndim=1,mode='c'] m,np.ndarray[np.float32_t,ndim=1,mode='c'] v,float pot=1.0,int nlat=-1,int nlon=-1)
        > weights a Gauss-Legendre map with the pixel sizes to a given power
        > input array will optionally be overwritten(!)
        > float precision

        input:
        - m          Gauss-Legendre map (note: len(m)==nlat*nlon)
        - v          pixel sizes        (note: len(v)==nlat)

        parameters:
        - p          power                                       (default: 1)
        - nlat       number of latitudinal bins or "rings"       (assumed default: len(v))
        - nlon       number of longitudinal bins                 (assumed default: 2*nlat-1)
        - overwrite  whether to overwrite the input array or not (default: False)

        output:
        - m XOR n    Gauss-Legendre map (note: len(n)==len(m))

    """
    if(nlat<0):
        nlat = <int> len(v)
    if(nlon<0):
        nlon = 2*nlat-1
    check_para4mlen(<int> len(m),&nlat,&nlon)
    cdef float* p_v = <float*> v.data
    cdef np.ndarray[np.float32_t,ndim=1,mode='c'] n
    cdef float* p_o
    if(overwrite):
        p_o = <float*> m.data
    else:
        n = np.copy(m)
        p_o = <float*> n.data
    cdef float vol
    cdef int lat,lon,ii = 0
    if(p==1):
        for lat in range(nlat):
            vol = p_v[lat]
            for lon in range(nlon):
                p_o[ii] *= vol
                ii += 1
    elif(p==-1):
        for lat in range(nlat):
            vol = 1/p_v[lat]
            for lon in range(nlon):
                p_o[ii] *= vol
                ii += 1
    elif(p<>0):
        for lat in range(nlat):
            vol = p_v[lat]**p
            for lon in range(nlon):
                p_o[ii] *= vol
                ii += 1
    if(overwrite):
        return m
    else:
        return n

##-----------------------------------------------------------------------------



##=============================================================================

cpdef np.ndarray[np.float64_t,ndim=1,mode='c'] anaalm(np.ndarray[np.complex128_t,ndim=1,mode='c'] a,int lmax=-1,int mmax=-1):
    """
        anaalm(np.ndarray[np.complex128_t,ndim=1,mode='c'] a,int lmax=-1,int mmax=-1):
        > computates the angular power spectrum of a given spherical harmonics transform

        input:
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

        parameters:
        - lmax  maximum l of the transform (assumed default obtained from len(a))
        - mmax  maximum m of the transform (assumed default: lmax)

        output:
        - c     angular power spectrum (note: len(c)==lmax+1)

    """
    if(lmax<0):
        if(mmax<0)or(mmax>lmax):
            lmax = (-3+int(sqrt(1+8*len(a))))/2
            mmax = lmax
        else:
            lmax = (<int> len(a)+(mmax+1)*mmax/2)/(mmax+1)-1
    elif(mmax<0)or(mmax>lmax):
            mmax = lmax
    check_para4alen(<int> len(a),&lmax,&mmax)
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] c = np.zeros(lmax+1,dtype=np.float64)
    cdef double* p_c = <double*> c.data
    cdef c128* p_a = <c128*> a.data
    cdef int ll,mm,ii = 0
    for ll in range(0,lmax+1):
        p_c[ll] += (p_a[ii].re*p_a[ii].re)/(2*ll+1)
        ii += 1
    for mm in range(1,mmax+1):
        for ll in range(mm,lmax+1):
            p_c[ll] += 2*(p_a[ii].re*p_a[ii].re+p_a[ii].im*p_a[ii].im)/(2*ll+1)
            ii += 1
    return c

##-----------------------------------------------------------------------------

def alm2map(np.ndarray[np.complex128_t,ndim=1,mode='c'] a,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,bint cl=False):
    """
        alm2map(np.ndarray[np.complex128_t,ndim=1,mode='c'] a,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,bint cl=False)
        > computates the Gauss-Legendre map from a given spherical harmonics transform

        input:
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

        parameters:
        - nlat  number of latitudinal bins or "rings"               (default: lmax+1)
        - nlon  number of longitudinal bins                         (default: 2*lmax+1)
        - lmax  maximum l of the transform                          (assumed default obtained from len(a))
        - mmax  maximum m of the transform                          (assumed default: lmax)
        - cl    whether to return the angular power spectrum or not (default: False)

        output:
        - m     Gauss-Legendre map     (note: len(m)==nlat*nlon)
        - c     angular power spectrum (note: len(c)==lmax+1)

    """
    if(nlat<0)or(nlon<0)or(lmax<0)or(mmax<0):
        set_para4alen(<int> len(a),&nlat,&nlon,&lmax,&mmax)
    else:
        check_para4alen(<int> len(a),&lmax,&mmax)
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] m = np.empty(nlat*nlon,dtype=np.float64)
    a2m_d(<c128*> a.data,lmax,mmax,<double*> m.data,nlat,nlon)
    if(cl):
        return m,anaalm(a,lmax,mmax)
    else:
        return m

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.complex128_t,ndim=1,mode='c'] map2alm(np.ndarray[np.float64_t,ndim=1,mode='c'] m,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1):
    """
        map2alm(np.ndarray[np.float64_t,ndim=1,mode='c'] m,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1)
        > computates the spherical harmonics transform of a given Gauss-Legendre map

        input:
        - m     Gauss-Legendre map (note: len(m)==nlat*nlon)

        parameters:
        - nlat  number of latitudinal bins or "rings" (assumed default obtained from len(m))
        - nlon  number of longitudinal bins           (assumed default obtained from len(m))
        - lmax  maximum l of the transform            (default: nlat-1)
        - mmax  maximum m of the transform            (default: lmax)

        output:
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

    """
    if(nlat<0)or(nlon<0)or(lmax<0)or(mmax<0):
        set_para4mlen(<int> len(m),&nlat,&nlon,&lmax,&mmax)
    else:
        check_para4mlen(<int> len(m),&nlat,&nlon)
    cdef np.ndarray[np.complex128_t,ndim=1,mode='c'] a = np.empty((lmax+1)*(mmax+1)-(mmax*(mmax+1))/2,dtype=np.complex128)
    m2a_d(<double*> m.data,nlat,nlon,<c128*> a.data,lmax,mmax,False)
    return a

##-----------------------------------------------------------------------------

def anafast(np.ndarray[np.float64_t,ndim=1,mode='c'] m,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,bint alm=False):
    """
        anafast(np.ndarray[np.float64_t,ndim=1,mode='c'] m,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,bint alm=False)
        > computates the angular power spectrum of a given Gauss-Legendre map

        input:
        - m     Gauss-Legendre map (note: len(m)==nlat*nlon)

        parameters:
        - nlat  number of latitudinal bins or "rings"               (assumed default obtained from len(m))
        - nlon  number of longitudinal bins                         (assumed default obtained from len(m))
        - lmax  maximum l of the transform                          (default: nlat-1)
        - mmax  maximum m of the transform                          (default: lmax)
        - alm   whether to return the transform additionally or not (default: False)

        output:
        - c     angular power spectrum        (note: len(c)==lmax+1)
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

    """
    if(nlat<0)or(nlon<0)or(lmax<0)or(mmax<0):
        set_para4mlen(<int> len(m),&nlat,&nlon,&lmax,&mmax)
    else:
        check_para4mlen(<int> len(m),&nlat,&nlon)
    cdef np.ndarray[np.complex128_t,ndim=1,mode='c'] a = np.empty((lmax+1)*(mmax+1)-(mmax*(mmax+1))/2,dtype=np.complex128)
    m2a_d(<double*> m.data,nlat,nlon,<c128*> a.data,lmax,mmax,False)
    if(alm):
        return anaalm(a,lmax,mmax),a
    else:
        return anaalm(a,lmax,mmax)

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.complex128_t,ndim=1,mode='c'] synalm(np.ndarray[np.float64_t,ndim=1,mode='c'] c,int lmax=-1,int mmax=-1):
    """
        synalm(np.ndarray[np.float64_t,ndim=1,mode='c'] c,int lmax=-1,int mmax=-1)
        > synthesises a spherical harmonics transform from a given angular power spectrum

        input:
        - c     angular power spectrum (note: len(c)==lmax+1)

        parameters:
        - lmax  maximum l of the transform (assumed default: len(c)-1)
        - mmax  maximum m of the transform (assumed default: lmax)

        output:
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

    """
    if(lmax<0)or(lmax>=(<int> len(c))):
        lmax = <int> len(c)-1
    if(mmax<0)or(mmax>lmax):
        mmax = lmax
    cdef np.ndarray[np.complex128_t,ndim=1,mode='c'] a = np.empty((lmax+1)*(mmax+1)-(mmax*(mmax+1))/2,dtype=np.complex128)
    cdef c128* p_a = <c128*> a.data
    cdef double* p_c = <double*> c.data
    cdef int ll,mm,ii = 0
    for ll in range(0,lmax+1):
        p_a[ii].re = np.random.normal(0,sqrt(p_c[ll])) ## alternative?
        p_a[ii].im = 0
        ii += 1
    for mm in range(1,mmax+1):
        for ll in range(mm,lmax+1):
            p_a[ii].re = np.random.normal(0,sqrt(0.5*p_c[ll])) ## alternative?
            p_a[ii].im = np.random.normal(0,sqrt(0.5*p_c[ll])) ## alternative?
            ii += 1
    return a

##-----------------------------------------------------------------------------

def synfast(np.ndarray[np.float64_t,ndim=1,mode='c'] c,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,alm=False):
    """
        synfast(np.ndarray[np.float64_t,ndim=1,mode='c'] c,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,alm=False)
        > synthesises a spherical harmonics transform from a given angular power spectrum

        input:
        - c     angular power spectrum (note: len(c)==lmax+1)

        parameters:
        - nlat  number of latitudinal bins or "rings"               (default: lmax+1)
        - nlon  number of longitudinal bins                         (default: 2*lmax+1)
        - lmax  maximum l of the transform                          (assumed default: len(c)-1)
        - mmax  maximum m of the transform                          (assumed default: lmax)
        - alm   whether to return the transform additionally or not (default: False)

        output:
        - m     Gauss-Legendre map            (note: len(m)==nlat*nlon)
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

    """
    if(lmax<0)or(lmax>=(<int> len(c))):
        lmax = <int> len(c)-1
    if(mmax<0)or(mmax>lmax):
        mmax = lmax
    cdef np.ndarray[np.complex128_t,ndim=1,mode='c'] a = synalm(c,lmax,mmax)
    if(nlat<0):
        nlat = lmax+1
    if(nlon<0):
        nlon = 2*lmax+1
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] m = alm2map(a,nlat,nlon,lmax,mmax,cl=False)
    if(alm):
        return m,a
    else:
        return m

##-----------------------------------------------------------------------------

cpdef double dotlm(np.ndarray[np.complex128_t,ndim=1,mode='c'] a, np.ndarray[np.complex128_t,ndim=1,mode='c'] b,int lmax=-1,int mmax=-1):
    """
        dotlm(np.ndarray[np.complex128_t,ndim=1,mode='c'] a, np.ndarray[np.complex128_t,ndim=1,mode='c'] b,int lmax=-1,int mmax=-1)
        > computates the inner product of two given spherical harmonics transform

        input:
        - a     spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)
        - b     spherical harmonics transform (note: len(b)==len(a))

        parameters:
        - lmax  maximum l of the transform (assumed default obtained from len(a))
        - mmax  maximum m of the transform (assumed default: lmax)

        output:
        - dot   inner product

    """
    if(lmax<0):
        if(mmax<0)or(mmax>lmax):
            lmax = (-3+int(sqrt(1+8*len(a))))/2
            mmax = lmax
        else:
            lmax = (<int> len(a)+(mmax+1)*mmax/2)/(mmax+1)-1
    elif(mmax<0)or(mmax>lmax):
            mmax = lmax
    check_para4alen(<int> len(a),&lmax,&mmax)
    check_para4alen(<int> len(b),&lmax,&mmax)
    cdef c128* p_a = <c128*> a.data
    cdef c128* p_b = <c128*> b.data
    cdef double dot = 0.0
    cdef int ll,mm,ii = 0
    for ll in range(0,lmax+1):
        dot += p_a[ii].re*p_b[ii].re
        ii += 1
    for mm in range(1,mmax+1):
        for ll in range(mm,lmax+1):
            dot += 2*(p_a[ii].re*p_b[ii].re+p_a[ii].im*p_b[ii].im)
            ii += 1
    return dot

##-----------------------------------------------------------------------------

cdef void gauk(int lmax,int mmax,double var,c128* p_a):
    """
        > multiplies a spherical harmonics transform with a Gaussian

    """
    cdef double k
    cdef int ll,mm
    for ll in range(lmax+1):
        k = exp(-0.5*ll*(ll+1)*var)
        for mm in range(min(mmax,ll)+1):
            i = ll+mm*lmax-mm*(mm-1)/2 ## cf. lm2i()
            p_a[i].re *= k
            p_a[i].im *= k

cpdef np.ndarray[np.complex128_t,ndim=1,mode='c'] smoothalm(np.ndarray[np.complex128_t,ndim=1,mode='c'] a,int lmax=-1,int mmax=-1,double fwhm=0.0,double sigma=-1.0,bint overwrite=True):
    """
        smoothalm(np.ndarray[np.complex128_t,ndim=1,mode='c'] a,int lmax=-1,int mmax=-1,double fwhm=0.0,double sigma=-1.0,bint overwrite=True)
        > smoothes a spherical harmonics transform with a Gaussian kernel
        > input array will optionally be overwritten(!)

        input:
        - a          spherical harmonics transform (note: len(a)==(lmax+1)*(mmax+1)-(mmax*(mmax+1))/2)

        parameters:
        - lmax       maximum l of the transform                  (assumed default obtained from len(a))
        - mmax       maximum m of the transform                  (assumed default: lmax)
        - fwhm       full width at half maximum of the Gaussian  (default: 0.0)
        - sigma      standard deviation of the Gaussian          (default: fwhm/(2*sqrt(2*log(2.0))))
        - overwrite  whether to overwrite the input array or not (default: True)

        output:
        - a XOR b    smoothed spherical harmonics transform (note: len(b)==len(a))

    """
    if(lmax<0):
        if(mmax<0)or(mmax>lmax):
            lmax = (-3+int(sqrt(1+8*len(a))))/2
            mmax = lmax
        else:
            lmax = (<int> len(a)+(mmax+1)*mmax/2)/(mmax+1)-1
    elif(mmax<0)or(mmax>lmax):
            mmax = lmax
    check_para4alen(<int> len(a),&lmax,&mmax)
    if(sigma<0):
        sigma = fwhm/(2*sqrt(2*log(2.0)))
    cdef np.ndarray[np.complex128_t,ndim=1,mode='c'] b
    if(overwrite):
        b = np.copy(a)
        gauk(lmax,mmax,sigma*sigma,<c128*> b.data)
        return b
    else:
        gauk(lmax,mmax,sigma*sigma,<c128*> a.data)
        return a

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.float64_t,ndim=1,mode='c'] smoothmap(np.ndarray[np.float64_t,ndim=1,mode='c'] m,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,double fwhm=0.0,double sigma=-1.0):
    """
        smoothmap(np.ndarray[np.float64_t,ndim=1,mode='c'] m,int nlat=-1,int nlon=-1,int lmax=-1,int mmax=-1,double fwhm=0.0,double sigma=-1.0,bint overwrite=False)
        > smoothes a Gauss-Legendre map with a Gaussian kernel

        input:
        - m      Gauss-Legendre map (note: len(m)==nlat*nlon)

        parameters:
        - nlat   number of latitudinal bins or "rings"      (assumed default obtained from len(m))
        - nlon   number of longitudinal bins                (assumed default obtained from len(m))
        - lmax   maximum l of the transform                 (default: nlat-1)
        - mmax   maximum m of the transform                 (default: lmax)
        - fwhm   full width at half maximum of the Gaussian (default: 0.0)
        - sigma  standard deviation of the Gaussian         (default: fwhm/(2*sqrt(2*log(2.0))))

        output:
        - n      smoothed Gauss-Legendre map (note: len(n)==len(m))

    """
    if(nlat<0)or(nlon<0)or(lmax<0)or(mmax<0):
        set_para4mlen(<int> len(m),&nlat,&nlon,&lmax,&mmax)
    else:
        check_para4mlen(<int> len(m),&nlat,&nlon)
    if(sigma<0):
        sigma = fwhm/(2*sqrt(2*log(2.0)))
    return alm2map(smoothalm(map2alm(m,lmax=lmax,mmax=mmax),lmax=lmax,mmax=mmax,sigma=sigma,overwrite=True),nlat=nlat,nlon=nlon,lmax=lmax,mmax=mmax,cl=False)

##-----------------------------------------------------------------------------

cdef void gauleg(int nlat,double* p_x,double* p_w):
    """
        gauleg(int nlat,double* p_x,double* p_w)
        > computes roots and weights of a Gauss-Legendre transformation over [-1,1]

    """
    cdef int i,j,imax = (nlat+1)/2
    cdef bint done
    cdef double c0,c1,p1,p2,p3,pp
    for i in range(1,imax+1):
        c0 = cos(pi*(i-0.25)/(nlat+0.5))
        done = False
        while(True):
            p1 = 1.0
            p2 = 0.0
            c1 = c0
            for j in range(1,nlat+1):
                p3 = p2
                p2 = p1
                p1 = ((2*j-1)*c0*p2-(j-1)*p3)/j
            pp = nlat*(c0*p1-p2)/(c0*c0-1)
            c0 = c1-p1/pp
            if(done):
                break
            if(fabs(c0-c1)<=3.0E-15): ## arbitrary small number
                done = True
        p_x[i-1] = -c0
        p_x[nlat-i] = c0
        p_w[i-1] = p_w[nlat-i] = 2/((1-c0*c0)*pp*pp)

cdef void gauleg_w2b(int nlat,double* p_x,double* p_b):
    """
        gauleg_w2b(int nlat,double* p_x,double* p_b)
        > computes the pixel cosine bounds given the roots

    """
    p_b[0] = -1.0
    gauleg(nlat,p_x,p_b+1)
    cdef int i
    for i in range(nlat/2):
        p_b[i+1] += p_b[i]
        p_b[nlat-i-1] = -p_b[i+1]
    p_b[nlat/2] = 0.0
    p_b[nlat] = 1.0

cdef int get_lat4bounds(int nlat,double x,double* p_b):
    """
        get_lat4bounds(int nlat,double x,double* p_b)
        > assignes latitudinal bin or "ring"

    """
    cdef int i
    for i in range(nlat):
        if(x>p_b[nlat-i-1]):
            break
    return i

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.float64_t,ndim=1,mode='c'] vol(int nlat,int nlon=-1):
    """
        vol(int nlat,int nlon=-1)
        > computates the pixel size for each latitudinal bin or "ring"

        input:
        - nlat  number of latitudinal bins or "rings"

        parameter:
        - nlon  number of longitudinal bins (default: 2*nlat-1)

        output:
        - v     pixel sizes (note: len(v)==nlat)

    """
    if(nlon<0):
        nlon = 2*nlat-1
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] v = np.empty(nlat,dtype=np.float64)
    cdef double* p_v = <double*> v.data
    gauleg(nlat,p_v,p_v)
    cdef int i
    for i in range(nlat):
        p_v[i] *= 2*pi/nlon
    return v

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.float64_t,ndim=1,mode='c'] weight(np.ndarray[np.float64_t,ndim=1,mode='c'] m,np.ndarray[np.float64_t,ndim=1,mode='c'] v,double p=1.0,int nlat=-1,int nlon=-1,bint overwrite=False):
    """
        weight(np.ndarray[np.float64_t,ndim=1,mode='c'] m,np.ndarray[np.float64_t,ndim=1,mode='c'] v,double pot=1.0,int nlat=-1,int nlon=-1,bint overwrite=False)
        > weights a Gauss-Legendre map with the pixel sizes to a given power
        > input array will optionally be overwritten(!)

        input:
        - m          Gauss-Legendre map (note: len(m)==nlat*nlon)
        - v          pixel sizes        (note: len(v)==nlat)

        parameters:
        - p          power                                       (default: 1)
        - nlat       number of latitudinal bins or "rings"       (assumed default: len(v))
        - nlon       number of longitudinal bins                 (assumed default: 2*nlat-1)
        - overwrite  whether to overwrite the input array or not (default: False)

        output:
        - m XOR n    Gauss-Legendre map (note: len(n)==len(m))

    """
    if(nlat<0):
        nlat = <int> len(v)
    if(nlon<0):
        nlon = 2*nlat-1
    check_para4mlen(<int> len(m),&nlat,&nlon)
    cdef double* p_v = <double*> v.data
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] n
    cdef double* p_o
    if(overwrite):
        p_o = <double*> m.data
    else:
        n = np.copy(m)
        p_o = <double*> n.data
    cdef double vol
    cdef int lat,lon,ii = 0
    if(p==1):
        for lat in range(nlat):
            vol = p_v[lat]
            for lon in range(nlon):
                p_o[ii] *= vol
                ii += 1
    elif(p==-1):
        for lat in range(nlat):
            vol = 1/p_v[lat]
            for lon in range(nlon):
                p_o[ii] *= vol
                ii += 1
    elif(p<>0):
        for lat in range(nlat):
            vol = p_v[lat]**p
            for lon in range(nlon):
                p_o[ii] *= vol
                ii += 1
    if(overwrite):
        return m
    else:
        return n

##-----------------------------------------------------------------------------

def bounds(int nlat,int nlon=-1):
    """
        bounds(int nlat,int nlon=-1)
        > computates the latitudinal and longitudinal bounds given a set of parameters of a Gauss-Legendre map

        input:
        - nlat  number of latitudinal bins or "rings"

        parameter:
        - nlon  number of longitudinal bins (default: 2*nlat-1)

        output:
        - x     longitudinal bounds in radian (note: len(x)==nlon+1)
        - y     latitudinal bounds in radian  (note: len(y)==nlat+1)

    """
    if(nlon<nlat):
        nlon = 2*nlat-1
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] x = np.empty(nlon+1,dtype=np.float64) ## first used as dummy
    cdef double* p_x = <double*> x.data
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] y = np.empty(nlat+1,dtype=np.float64)
    cdef double* p_y = <double*> y.data
    p_x[0] = -1.0
    p_y[0] = 0.0
    gauleg(nlat,p_x+1,p_x+1)
    cdef int i
    for i in range(nlat/2):
        p_x[i+1] += p_x[i]
        p_y[i+1] = acos(-p_x[i+1])
        p_y[nlat-i-1] = acos(p_x[i+1])
    p_y[nlat/2] = pi/2
    p_y[nlat] = pi
    p_x[0] = 0.0
    for i in range(nlon):
        p_x[i+1] = p_x[i]+2*pi/nlon
    return x,y

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.float64_t,ndim=1,mode='c'] pix2ang(int pix,int nlat,int nlon=-1):
    """
        pix2ang(int pix,int nlat,int nlon=-1)
        > computates the [phi,theta] vector for a given pixel index

        input:
        - pix   pixel index
        - nlat  number of latitudinal bins or "rings"

        parameter:
        - nlon  number of longitudinal bins (defalut: 2*nlat-1)

        output:
        - ang   vector containing [phi,theta] in radian

    """
    if(nlon<0):
        nlon = 2*nlat-1
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] x = np.empty(nlat,dtype=np.float64)
    cdef double* p_x = <double*> x.data
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] b = np.empty(nlat+1,dtype=np.float64)
    cdef double* p_b = <double*> b.data
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] ang = np.empty(2,dtype=np.float64)
    cdef double* p_ang = <double*> ang.data
    gauleg_w2b(nlat,p_x,p_b)
    p_ang[0] = 2*pi*(pix%nlon+0.5)/nlon
    p_ang[1] = acos(p_x[nlat-1-pix/nlon])
    if(p_ang[1]<0):
        p_ang[1] += 2*pi
    return ang

cpdef int ang2pix(np.ndarray[np.float64_t,ndim=1,mode='c'] ang,int nlat,int nlon=-1):
    """
        ang2pix(np.ndarray[np.float64_t,ndim=1,mode='c'] ang,int nlat,int nlon=-1)
        > computates the pixel index for a given [phi,theta] vector

        input:
        - ang   vector containing [phi,theta] in radian
        - nlat  number of latitudinal bins or "rings"

        parameter:
        - nlon  number of longitudinal bins (defalut: 2*nlat-1)

        output:
        - pix   pixel index

    """
    if(nlon<0):
        nlon = 2*nlat-1
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] x = np.empty(nlat,dtype=np.float64)
    cdef double* p_x = <double*> x.data
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] b = np.empty(nlat+1,dtype=np.float64)
    cdef double* p_b = <double*> b.data
    cdef double* p_ang = <double*> ang.data
    gauleg_w2b(nlat,p_x,p_b)
    return int((nlon*p_ang[0]/(2*pi))%nlon+nlon*get_lat4bounds(nlat,cos(p_ang[1]),p_b))

##-----------------------------------------------------------------------------

cpdef np.ndarray[np.float64_t,ndim=1,mode='c'] ang2xyz(np.ndarray[np.float64_t,ndim=1,mode='c'] ang):
    """
        ang2xyz(np.ndarray[np.float64_t,ndim=1,mode='c'] ang)
        > computates the [x,y,z] vector for a given [phi,theta] vector

        input:
        - ang  vector containing [phi,theta] in radian

        output:
        - xyz  vector containing [x,y,z]

    """
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] xyz = np.empty(3,dtype=np.float64)
    cdef double* p_xyz = <double*> xyz.data
    cdef double* p_ang = <double*> ang.data
    p_xyz[0] = sin(p_ang[1])*cos(p_ang[0])
    p_xyz[1] = sin(p_ang[1])*sin(p_ang[0])
    p_xyz[2] = cos(p_ang[1])
    return xyz

cpdef np.ndarray[np.float64_t,ndim=1,mode='c'] xyz2ang(np.ndarray[np.float64_t,ndim=1,mode='c'] xyz):
    """
        xyz2ang(np.ndarray[np.float64_t,ndim=1,mode='c'] xyz)
        > computates the [phi,theta] vector for a given [x,y,z] vector

        input:
        - xyz  vector containing [x,y,z]

        output:
        - ang  vector containing [phi,theta] in radian

    """
    cdef np.ndarray[np.float64_t,ndim=1,mode='c'] ang = np.empty(2,dtype=np.float64)
    cdef double* p_ang = <double*> ang.data
    cdef double* p_xyz = <double*> xyz.data
    cdef double r = 0.0
    cdef int i
    for i in range(3):
        r += p_xyz[i]*p_xyz[i]
    r = sqrt(r)
    p_ang[1] = acos(p_xyz[2]/r)
    p_ang[0] = atan2(p_xyz[1],p_xyz[0])
    if(p_ang[0]<0):
        p_ang[0] += 2*pi
    return ang


##-----------------------------------------------------------------------------

cpdef np.ndarray[np.float64_t,ndim=1,mode='c'] pix2xyz(int pix,int nlat,int nlon=-1):
    """
        pix2xyz(int pix,int nlat,int nlon=-1)
        > computates the [x,y,z] vector for a given pixel index

        input:
        - pix   pixel index
        - nlat  number of latitudinal bins or "rings"

        parameter:
        - nlon  number of longitudinal bins (defalut: 2*nlat-1)

        output:
        - xyz  vector containing [x,y,z]

    """
    return ang2xyz(pix2ang(pix,nlat,nlon))


cpdef int xyz2pix(np.ndarray[np.float64_t,ndim=1,mode='c'] xyz,int nlat,int nlon=-1):
    """
        xyz2pix(np.ndarray[np.float64_t,ndim=1,mode='c'] xyz,int nlat,int nlon=-1)
        > computates the pixel index for a given [x,y,z] vector

        input:
        - xyz  vector containing [x,y,z]
        - nlat  number of latitudinal bins or "rings"

        parameter:
        - nlon  number of longitudinal bins (defalut: 2*nlat-1)

        output:
        - pix   pixel index

    """
    return ang2pix(xyz2ang(xyz),nlat,nlon)

##=============================================================================


