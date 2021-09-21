/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid 

Source file: lib/qcd/spin/Dirac.h

Copyright (C) 2015
Copyright (C) 2016

Author: Antonin Portelli <antonin.portelli@me.com>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: Peter Boyle <peterboyle@Peters-MacBook-Pro-2.local>
Author: paboyle <paboyle@ph.ed.ac.uk>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
			   /*  END LEGAL */
#ifndef GRID_QCD_DIRAC_H
#define GRID_QCD_DIRAC_H

			   // Gamma matrices using the code generated by the Mathematica notebook 
			   // gamma-gen/gamma-gen.nb in Gamma.cc & Gamma.h
			   ////////////////////////////////////////////////////////////////////////////////
#include <Grid/qcd/spin/Gamma.h>

NAMESPACE_BEGIN(Grid);

// Dirac algebra adjoint operator (not in  to overload other adj)
inline Gamma adj(const Gamma &g)
{
  return Gamma (Gamma::adj[g.g]);
}



// Dirac algebra mutliplication operator
inline Gamma operator*(const Gamma &g1, const Gamma &g2)
{
  return Gamma (Gamma::mul[g1.g][g2.g]);
}

// general left multiply
template<class vtype> 
accelerator_inline auto operator*(const Gamma &G, const iScalar<vtype> &arg)
  ->typename std::enable_if<matchGridTensorIndex<iScalar<vtype>,SpinorIndex>::notvalue,iScalar<vtype>>::type 
{
  iScalar<vtype> ret;
  ret._internal=G*arg._internal;
  return ret;
}

template<class vtype,int N>
accelerator_inline auto operator*(const Gamma &G, const iVector<vtype, N> &arg)
  ->typename std::enable_if<matchGridTensorIndex<iVector<vtype,N>,SpinorIndex>::notvalue,iVector<vtype,N>>::type 
{
  iVector<vtype,N> ret;
  for(int i=0;i<N;i++){
    ret._internal[i]=G*arg._internal[i];
  }
  return ret;
}

template<class vtype, int N>
accelerator_inline auto operator*(const Gamma &G, const iMatrix<vtype, N> &arg) 
  ->typename std::enable_if<matchGridTensorIndex<iMatrix<vtype,N>,SpinorIndex>::notvalue,iMatrix<vtype,N>>::type 
{
  iMatrix<vtype,N> ret;
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      ret._internal[i][j]=G*arg._internal[i][j];
    }}
  return ret;
}

// general right multiply
template<class vtype>
accelerator_inline auto operator*(const iScalar<vtype> &arg, const Gamma &G)
  ->typename std::enable_if<matchGridTensorIndex<iScalar<vtype>,SpinorIndex>::notvalue,iScalar<vtype>>::type 
{
  iScalar<vtype> ret;
  ret._internal=arg._internal*G;
  return ret;
}

template<class vtype, int N>
accelerator_inline auto operator * (const iMatrix<vtype, N> &arg, const Gamma &G) 
  ->typename std::enable_if<matchGridTensorIndex<iMatrix<vtype,N>,SpinorIndex>::notvalue,iMatrix<vtype,N>>::type 
{
  iMatrix<vtype,N> ret;
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      ret._internal[i][j]=arg._internal[i][j]*G;
    }}
  return ret;
}

// Gamma-left matrices gL_mu = g_mu*(1 - g5)
////////////////////////////////////////////////////////////////////////////////
class GammaL
{
public:
  typedef Gamma::Algebra Algebra;
  Gamma gamma;
public:
  GammaL(const Algebra initg): gamma(initg) {}
  GammaL(const Gamma   initg): gamma(initg) {}
};

// vector multiply
template<class vtype> 
accelerator_inline auto operator*(const GammaL &gl, const iVector<vtype, Ns> &arg)
  ->typename std::enable_if<matchGridTensorIndex<iVector<vtype, Ns>, SpinorIndex>::value, iVector<vtype, Ns>>::type
{
  iVector<vtype, Ns> buf;
  
  buf(0) = 0.;
  buf(1) = 0.;
  buf(2) = 2.*arg(2);
  buf(3) = 2.*arg(3);
  
  return gl.gamma*buf;
};

// matrix left multiply
template<class vtype> 
accelerator_inline auto operator*(const GammaL &gl, const iMatrix<vtype, Ns> &arg)
  ->typename std::enable_if<matchGridTensorIndex<iMatrix<vtype, Ns>, SpinorIndex>::value, iMatrix<vtype, Ns>>::type
{
  iMatrix<vtype, Ns> buf;
  
  for(unsigned int i = 0; i < Ns; ++i)
    {
      buf(0, i) = 0.;
      buf(1, i) = 0.;
      buf(2, i) = 2.*arg(2, i);
      buf(3, i) = 2.*arg(3, i);
    }
  
  return gl.gamma*buf;
};

// matrix right multiply
template<class vtype> 
accelerator_inline auto operator*(const iMatrix<vtype, Ns> &arg, const GammaL &gl)
  ->typename std::enable_if<matchGridTensorIndex<iMatrix<vtype, Ns>, SpinorIndex>::value, iMatrix<vtype, Ns>>::type
{
  iMatrix<vtype, Ns> buf;
  
  buf = arg*gl.gamma;
  for(unsigned int i = 0; i < Ns; ++i)
    {
      buf(i, 0) = 0.;
      buf(i, 1) = 0.;
      buf(i, 2) = 2.*buf(i, 2);
      buf(i, 3) = 2.*buf(i, 3);
    }
  
  return buf;
};

//general left multiply
template<class vtype> 
accelerator_inline auto operator*(const GammaL &gl, const iScalar<vtype> &arg)
  ->typename std::enable_if<matchGridTensorIndex<iScalar<vtype>,SpinorIndex>::notvalue,iScalar<vtype>>::type 
{
  iScalar<vtype> ret;
  ret._internal=gl*arg._internal;
  return ret;
}

template<class vtype,int N>
accelerator_inline auto operator*(const GammaL &gl, const iVector<vtype, N> &arg)
  ->typename std::enable_if<matchGridTensorIndex<iVector<vtype,N>,SpinorIndex>::notvalue,iVector<vtype,N>>::type 
{
  iVector<vtype,N> ret;
  for(int i=0;i<N;i++){
    ret._internal[i]=gl*arg._internal[i];
  }
  return ret;
}

template<class vtype, int N>
accelerator_inline auto operator*(const GammaL &gl, const iMatrix<vtype, N> &arg) 
  ->typename std::enable_if<matchGridTensorIndex<iMatrix<vtype,N>,SpinorIndex>::notvalue,iMatrix<vtype,N>>::type 
{
  iMatrix<vtype,N> ret;
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      ret._internal[i][j]=gl*arg._internal[i][j];
    }}
  return ret;
}

//general right multiply
template<class vtype>
accelerator_inline auto operator*(const iScalar<vtype> &arg, const GammaL &gl)
  ->typename std::enable_if<matchGridTensorIndex<iScalar<vtype>,SpinorIndex>::notvalue,iScalar<vtype>>::type 
{
  iScalar<vtype> ret;
  ret._internal=arg._internal*gl;
  return ret;
}

template<class vtype, int N>
accelerator_inline auto operator * (const iMatrix<vtype, N> &arg, const GammaL &gl) 
  ->typename std::enable_if<matchGridTensorIndex<iMatrix<vtype,N>,SpinorIndex>::notvalue,iMatrix<vtype,N>>::type 
{
  iMatrix<vtype,N> ret;
  for(int i=0;i<N;i++){
    for(int j=0;j<N;j++){
      ret._internal[i][j]=arg._internal[i][j]*gl;
    }}
  return ret;
}

NAMESPACE_END(Grid);

#endif
