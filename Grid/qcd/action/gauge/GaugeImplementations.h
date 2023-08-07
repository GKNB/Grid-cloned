/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/action/gauge/GaugeImplementations.h

Copyright (C) 2015

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

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
			   /*  END LEGAL */
#ifndef GRID_QCD_GAUGE_IMPLEMENTATIONS_H
#define GRID_QCD_GAUGE_IMPLEMENTATIONS_H

#include "GaugeImplTypes.h"

NAMESPACE_BEGIN(Grid);

// Composition with smeared link, bc's etc.. probably need multiple inheritance
// Variable precision "S" and variable Nc
template <class GimplTypes> class PeriodicGaugeImpl : public GimplTypes {
public:
  INHERIT_GIMPL_TYPES(GimplTypes);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Support needed for the assembly of loops including all boundary condition
  // effects such as Conjugate bcs
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////

  template <class covariant>
  static inline Lattice<covariant>
  CovShiftForward(const GaugeLinkField &Link, int mu,
                  const Lattice<covariant> &field) {
    return PeriodicBC::CovShiftForward(Link, mu, field);
  }

  template <class covariant>
  static inline Lattice<covariant>
  CovShiftBackward(const GaugeLinkField &Link, int mu,
                   const Lattice<covariant> &field) {
    return PeriodicBC::CovShiftBackward(Link, mu, field);
  }
  static inline GaugeLinkField
  CovShiftIdentityBackward(const GaugeLinkField &Link, int mu) {
    return PeriodicBC::CovShiftIdentityBackward(Link, mu);
  }
  static inline GaugeLinkField
  CovShiftIdentityForward(const GaugeLinkField &Link, int mu) {
    return PeriodicBC::CovShiftIdentityForward(Link,mu);
  }
  static inline GaugeLinkField ShiftStaple(const GaugeLinkField &Link, int mu) {
    return PeriodicBC::ShiftStaple(Link,mu);
  }

  //Same as Cshift for periodic BCs
  static inline GaugeLinkField CshiftLink(const GaugeLinkField &Link, int mu, int shift){
    return PeriodicBC::CshiftLink(Link,mu,shift);
  }

  static inline bool isPeriodicGaugeField(void) { return true; }

  //Apply the Fourier acceleration to the gauge fixing step
  //The form of the FFT and the eigenvalues of the Laplacian are BC-dependent
  static inline void FourierAcceleratedGfix(GaugeLinkField &dmuAmu, int orthog){
    GridBase* grid = dmuAmu.Grid();
    GaugeLinkField dmuAmu_p(grid);

    LatticeComplex  Fp(grid), one(grid), psq(grid);
    one = Complex(1.0,0.0); 

    FFT theFFT((GridCartesian *)grid);

    std::vector<int> mask(Nd,1);
    for(int mu=0;mu<Nd;mu++) if (mu==orthog) mask[mu]=0;
    theFFT.FFT_dim_mask(dmuAmu_p,dmuAmu,mask,FFT::forward);

    //////////////////////////////////
    // Work out Fp = psq_max/ psq...
    // Avoid singularities in Fp
    //////////////////////////////////
    Coordinate latt_size = grid->GlobalDimensions();
    Coordinate coor(grid->_ndimension,0);
    LatticeComplex  pmu(grid); 

    psq = Zero();
    for(int mu=0;mu<Nd;mu++) {
      if ( mu != orthog ) { 
	Real TwoPiL =  M_PI * 2.0/ latt_size[mu];
	LatticeCoordinate(pmu,mu);
	pmu = TwoPiL * pmu ;
	psq = psq + 4.0*sin(pmu*0.5)*sin(pmu*0.5); 
      }
    }

    Complex psqMax(16.0);
    Fp =  psqMax*one/psq;

    pokeSite(TComplex(16.0),Fp,coor);
    if( (orthog>=0) && (orthog<Nd) ){
      for(int t=0;t<grid->GlobalDimensions()[orthog];t++){
	coor[orthog]=t;
	pokeSite(TComplex(16.0),Fp,coor);
      }
    }
    
    dmuAmu_p  = dmuAmu_p * Fp; 

    theFFT.FFT_dim_mask(dmuAmu,dmuAmu_p,mask,FFT::backward);
  }


};

// Composition with smeared link, bc's etc.. probably need multiple inheritance
// Variable precision "S" and variable Nc
class ConjugateGaugeImplBase {
protected:
  static std::vector<int> _conjDirs;
};

  template <class GimplTypes> class ConjugateGaugeImpl : public GimplTypes, ConjugateGaugeImplBase {
private:
public:
  INHERIT_GIMPL_TYPES(GimplTypes);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Support needed for the assembly of loops including all boundary condition
  // effects such as Gparity.
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template <class covariant>
  static Lattice<covariant> CovShiftForward(const GaugeLinkField &Link, int mu,
                                            const Lattice<covariant> &field)
  {
    assert(_conjDirs.size() == Nd);
    if(_conjDirs[mu]) 
      return ConjugateBC::CovShiftForward(Link, mu, field);
    else
      return PeriodicBC::CovShiftForward(Link, mu, field);
  }

  template <class covariant>
  static Lattice<covariant> CovShiftBackward(const GaugeLinkField &Link, int mu,
                                             const Lattice<covariant> &field)
  {
    assert(_conjDirs.size() == Nd);
    if(_conjDirs[mu]) 
      return ConjugateBC::CovShiftBackward(Link, mu, field);
    else 
      return PeriodicBC::CovShiftBackward(Link, mu, field);
  }

  //If mu is a conjugate BC direction
  //Out(x) = U^dag_\mu(x-mu)  | x_\mu != 0
  //       = U^T_\mu(L-1)  | x_\mu == 0
  //else
  //Out(x) = U^dag_\mu(x-mu mod L)
  static inline GaugeLinkField
  CovShiftIdentityBackward(const GaugeLinkField &Link, int mu)
  {
    assert(_conjDirs.size() == Nd);
    if(_conjDirs[mu]) 
      return ConjugateBC::CovShiftIdentityBackward(Link, mu);
    else 
      return PeriodicBC::CovShiftIdentityBackward(Link, mu);
  }
  static inline GaugeLinkField
  CovShiftIdentityForward(const GaugeLinkField &Link, int mu)
  {
    assert(_conjDirs.size() == Nd);
    if(_conjDirs[mu]) 
      return ConjugateBC::CovShiftIdentityForward(Link,mu);
    else
      return PeriodicBC::CovShiftIdentityForward(Link,mu);
  }


  //If mu is a conjugate BC direction
  //Out(x) = S_\mu(x+mu)  | x_\mu != L-1
  //       = S*_\mu(x+mu)  | x_\mu == L-1
  //else
  //Out(x) = S_\mu(x+mu mod L)
  //Note: While this is used for Staples it is also applicable for shifting gauge links or gauge transformation matrices
  static inline GaugeLinkField ShiftStaple(const GaugeLinkField &Link, int mu)
  {
    assert(_conjDirs.size() == Nd);
    if(_conjDirs[mu]) 
      return ConjugateBC::ShiftStaple(Link,mu);
    else     
      return PeriodicBC::ShiftStaple(Link,mu);
  }

  //Boundary-aware C-shift of gauge links / gauge transformation matrices
  //For conjugate BC direction
  //shift = 1
  //Out(x) = U_\mu(x+\hat\mu)  | x_\mu != L-1
  //       = U*_\mu(0)  | x_\mu == L-1
  //shift = -1
  //Out(x) = U_\mu(x-mu)  | x_\mu != 0
  //       = U*_\mu(L-1)  | x_\mu == 0
  //else
  //shift = 1
  //Out(x) = U_\mu(x+\hat\mu mod L)
  //shift = -1
  //Out(x) = U_\mu(x-\hat\mu mod L)
  static inline GaugeLinkField CshiftLink(const GaugeLinkField &Link, int mu, int shift){
    assert(_conjDirs.size() == Nd);
    if(_conjDirs[mu]) 
      return ConjugateBC::CshiftLink(Link,mu,shift);
    else     
      return PeriodicBC::CshiftLink(Link,mu,shift);
  }

  static inline void       setDirections(const std::vector<int> &conjDirs) { _conjDirs=conjDirs; }
  static inline std::vector<int> getDirections(void) { return _conjDirs; }
  static inline bool isPeriodicGaugeField(void) { return false; }

  //Apply the Fourier acceleration to the gauge fixing step
  //The form of the FFT and the eigenvalues of the Laplacian are BC-dependent
  static inline void FourierAcceleratedGfix(GaugeLinkField &dmuAmu, int orthog){
    int nconjdir = 0;
    for(int mu=0;mu<Nd;mu++)
      if ( mu != orthog && _conjDirs[mu] )
	nconjdir++;
    
    if(nconjdir == 0) return PeriodicGaugeImpl<GimplTypes>::FourierAcceleratedGfix(dmuAmu,orthog); //prd implementation is cheaper, plus you would get an unwanted pole in the momenta

    GridBase* grid = dmuAmu.Grid();
    Coordinate latt_size = grid->GlobalDimensions();

    FFT theFFT((GridCartesian *)grid);

    //In cconj directions, the real part of A_mu is antiperiodic and the imaginary part is periodic
    //To ensure the appropriate FFT basis, p=(2n+1)\pi/L, for the real part we must shift the momenta
    GaugeLinkField dmuAmu_r = real(dmuAmu), dmuAmu_i = imag(dmuAmu);
    
    std::vector<int> mask(Nd,1);
    for(int mu=0;mu<Nd;mu++) if (mu==orthog) mask[mu]=0;

    GaugeLinkField dmuAmu_pr(grid), dmuAmu_pi(grid);
    LatticeComplex  pmu(grid), shift(grid), Fp(grid), one(grid), psq(grid);
    one = Complex(1.0,0.0);
    Complex psqMax(16.0);

    { //Imaginary part, periodic BCs in all directions
      theFFT.FFT_dim_mask(dmuAmu_pi,dmuAmu_i,mask,FFT::forward);

      psq = Zero();
      for(int mu=0;mu<Nd;mu++) {
	if ( mu != orthog ) { 
	  Real TwoPiL =  M_PI * 2.0/ latt_size[mu];
	  LatticeCoordinate(pmu,mu);
	  pmu = TwoPiL * pmu ;
	  psq = psq + 4.0*sin(pmu*0.5)*sin(pmu*0.5); 
	}
      }

      Fp =  psqMax*one/psq;

      Coordinate coor(grid->_ndimension,0);
      pokeSite(TComplex(16.0),Fp,coor);
      if( (orthog>=0) && (orthog<Nd) ){
	for(int t=0;t<grid->GlobalDimensions()[orthog];t++){
	  coor[orthog]=t;
	  pokeSite(TComplex(16.0),Fp,coor);
	}
      }
    
      dmuAmu_pi  = dmuAmu_pi * Fp; 

      theFFT.FFT_dim_mask(dmuAmu_i,dmuAmu_pi,mask,FFT::backward);
    }

    { //Real part, antiperiodic BCs in cconj dirs, periodic otherwise
      //include phase factor so we are FFT'ing with the right momentum basis  p = (2k + 1)pi/L  for cconj directions   (  2pi/L otherwise )
      LatticeComplex  rphase = one, tmp(grid); 
      for(int mu=0;mu<Nd;mu++){
	if( mu != orthog && _conjDirs[mu] ){
	  LatticeCoordinate(tmp,mu);
	  tmp = tmp * Complex(0, -M_PI/latt_size[mu]);
	  tmp = exp(tmp);
	  rphase = rphase * tmp;
	}
      }
      dmuAmu_r = dmuAmu_r * rphase;

      theFFT.FFT_dim_mask(dmuAmu_pr,dmuAmu_r,mask,FFT::forward);

      psq = Zero();
      for(int mu=0;mu<Nd;mu++) {
	if ( mu != orthog ) { 
	  if( ! _conjDirs[mu] ){
	    Real TwoPiL =  M_PI * 2.0/ latt_size[mu];
	    LatticeCoordinate(pmu,mu);
	    pmu = TwoPiL * pmu ;
	    psq = psq + 4.0*sin(pmu*0.5)*sin(pmu*0.5); 
	  }else{
	    Real Pi2L =  M_PI / ( 2.0 * latt_size[mu] );
	    LatticeCoordinate(pmu,mu);
	    pmu = 2.0 * Pi2L * pmu ;
	    shift = Pi2L;
	    pmu = pmu + shift;

	    psq = psq + 4.0*sin(pmu)*sin(pmu); 
	  }
	}
      }

      //Note, psq=0 is not possible, hence we do not need to deal with the singularity (for some reason, if you do it anyway, it causes the plaquette to drift!)
      Fp =  psqMax*one/psq;
   
      dmuAmu_pr  = dmuAmu_pr * Fp; 

      theFFT.FFT_dim_mask(dmuAmu_r,dmuAmu_pr,mask,FFT::backward);
      dmuAmu_r = conjugate(rphase) * dmuAmu_r; //unapply the phase field
    }

    dmuAmu = dmuAmu_r + Complex(0.,1.)*dmuAmu_i;

  }

};

typedef PeriodicGaugeImpl<GimplTypesR> PeriodicGimplR; // Real.. whichever prec
typedef PeriodicGaugeImpl<GimplTypesF> PeriodicGimplF; // Float
typedef PeriodicGaugeImpl<GimplTypesD> PeriodicGimplD; // Double

typedef PeriodicGaugeImpl<GimplAdjointTypesR> PeriodicGimplAdjR; // Real.. whichever prec
typedef PeriodicGaugeImpl<GimplAdjointTypesF> PeriodicGimplAdjF; // Float
typedef PeriodicGaugeImpl<GimplAdjointTypesD> PeriodicGimplAdjD; // Double

typedef ConjugateGaugeImpl<GimplTypesR> ConjugateGimplR; // Real.. whichever prec
typedef ConjugateGaugeImpl<GimplTypesF> ConjugateGimplF; // Float
typedef ConjugateGaugeImpl<GimplTypesD> ConjugateGimplD; // Double

NAMESPACE_END(Grid);

#endif
