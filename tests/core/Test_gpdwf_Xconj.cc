    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/core/Test_gpdwf_Xconj.cc

    Copyright (C) 2015

Author: Christopher Kelly <ckelly@bnl.gov>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>

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

/**
 * Tests for the implementation of X-conjugate BCs
 *
 */

#include <Grid/Grid.h>

using namespace std;
using namespace Grid;

//A test implementation of the X-conjugate action as a wrapper around the regular 2f G-parity action
template<typename GPaction>
class XconjugateDWF{
public:
  typedef typename GPaction::FermionField GPfermion;
  typedef LatticeFermionD FermionField;
  GPaction *action;
  bool Xbar;
  XconjugateDWF(GPaction *action, bool Xbar = false): action(action), Xbar(Xbar){}

  template<typename Op>
  LatticeFermionD op11(const LatticeFermionD &in, const Op &op){
    GPfermion tmp1(in.Grid()), tmp2(in.Grid());
    tmp1.Checkerboard() = in.Checkerboard();
    tmp1 = Zero();
    PokeIndex<GparityFlavourIndex>(tmp1, in, 0);
    op(tmp2, tmp1);
    return PeekIndex<GparityFlavourIndex>(tmp2,0);
  }
  template<typename Op>
  LatticeFermionD op12(const LatticeFermionD &in, const Op &op){
    GPfermion tmp1(in.Grid()), tmp2(in.Grid());
    tmp1.Checkerboard() = in.Checkerboard();
    tmp1 = Zero();
    PokeIndex<GparityFlavourIndex>(tmp1, in, 1);
    op(tmp2, tmp1);
    return PeekIndex<GparityFlavourIndex>(tmp2,0);
  }
  
  template<typename Op>
  LatticeFermionD opFull(const LatticeFermionD &in, const Op &op){
    static Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
    static Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
    static Gamma X = C*g5;
    LatticeFermionD tmp1(in.Grid());
    tmp1.Checkerboard() = in.Checkerboard();
    tmp1 = -(X*conjugate(in));
    if(Xbar) tmp1 = -tmp1;
    
    LatticeFermionD out11 = op11(in, op);
    LatticeFermionD out12 = op12(tmp1, op);
    LatticeFermionD out = out11 + out12;
    return out;
  }

#define DEFOP(OP)  void OP(const FermionField &in, FermionField &out){ out=opFull(in, [&](GPfermion &out, const GPfermion &in){ return action->OP(in,out); }); }
  DEFOP(M);
  DEFOP(Mdag);
  DEFOP(Meooe);
  DEFOP(MeooeDag);
  DEFOP(Mooee);
  DEFOP(MooeeDag);
  DEFOP(MooeeInv);
  DEFOP(MooeeInvDag);
#undef DEFOP;
};

//A 2-flavor representation of the X-conjugate action acting on X-conjugate fermion fields
template<typename XconjAction, typename GPAction>
class Xconjugate2fWrapper{
public:
  typedef typename XconjAction::FermionField OneFlavorFermionField;
  typedef Lattice<iGparitySpinColourVector<typename OneFlavorFermionField::vector_type> > FermionField;

  XconjAction *action;
  GPAction *gaction;

  Xconjugate2fWrapper(XconjAction *action, GPAction *gaction): action(action), gaction(gaction){}
  
  template<typename Op>
  void opFull(FermionField &out, const FermionField &in, const Op &op){
    static Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
    static Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
    static Gamma X = C*g5;

    OneFlavorFermionField tmp(in.Grid());
    op(tmp, PeekIndex<GparityFlavourIndex>(in,0));
    
    out.Checkerboard() = tmp.Checkerboard();
    PokeIndex<GparityFlavourIndex>(out, tmp, 0);
    tmp = -(X*conjugate(tmp));
    tmp = tmp * action->Params.boundary_phase;
    PokeIndex<GparityFlavourIndex>(out, tmp, 1);
  }

#define DEFOP(OP)  void OP(const FermionField &in, FermionField &out){ opFull(out, in, \
									      [&](OneFlavorFermionField &out1f, \
										  const OneFlavorFermionField &in1f){ \
										return action->OP(in1f,out1f); \
									      }	\
									      ); \
                                                                      } 
  DEFOP(M);
  DEFOP(Mdag);
  DEFOP(Meooe);
  DEFOP(MeooeDag);
  DEFOP(Mooee);
  DEFOP(MooeeDag);
  DEFOP(MooeeInv);
  DEFOP(MooeeInvDag);
#undef DEFOP;
};

const Gamma & Xmatrix(){
  static Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
  static Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
  static Gamma X = C*g5;
  return X;
}

typedef typename GparityMobiusFermionR::FermionField FermionField2f;
typedef typename XconjugateMobiusFermionR::FermionField FermionField1f;


void boostXconjToGparity(FermionField2f &out, const FermionField1f &in){
  PokeIndex<GparityFlavourIndex>(out, in, 0);
  FermionField1f tmp = -(Xmatrix()*conjugate(in));
  PokeIndex<GparityFlavourIndex>(out, tmp, 1);
}
void boostXbarConjToGparity(FermionField2f &out, const FermionField1f &in){
  PokeIndex<GparityFlavourIndex>(out, in, 0);
  FermionField1f tmp = Xmatrix()*conjugate(in);
  PokeIndex<GparityFlavourIndex>(out, tmp, 1);
}

template<typename Field>
inline RealD norm2Diff(const Field &a, const Field &b){
  return norm2(Field(a-b));
}

void norm2DiffXconj(double &diff_f0, double &diff_f1, const FermionField2f &gp, const FermionField1f &xconj){
  FermionField1f tmp = PeekIndex<GparityFlavourIndex>(gp,0);
  diff_f0 = norm2(FermionField1f(tmp-xconj));
  tmp = PeekIndex<GparityFlavourIndex>(gp,1);
  FermionField1f xconj_f1 = -(Xmatrix()*conjugate(xconj));
  diff_f1 = norm2(FermionField1f(tmp-xconj_f1));
}
void norm2DiffXbarConj(double &diff_f0, double &diff_f1, const FermionField2f &gp, const FermionField1f &xconj){
  FermionField1f tmp = PeekIndex<GparityFlavourIndex>(gp,0);
  diff_f0 = norm2(FermionField1f(tmp-xconj));
  tmp = PeekIndex<GparityFlavourIndex>(gp,1);
  FermionField1f xconj_f1 = Xmatrix()*conjugate(xconj);
  diff_f1 = norm2(FermionField1f(tmp-xconj_f1));
}


int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  const int Ls=8;

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  GridParallelRNG          RNG5(FGrid);  RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG          RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);
  GridParallelRNG          RNG5rb(FrbGrid);  RNG5.SeedFixedIntegers(seeds5);

  LatticeGaugeField Umu(UGrid); 
  SU<Nc>::HotConfiguration(RNG4, Umu);

  Gamma X = Xmatrix();
  
  //Set up a regular MDWF action instance as well as X-conj and Xbar-conj versions
  RealD mass=0.01;
  RealD M5=1.8;
  RealD mob_b=1.5;
  GparityMobiusFermionD ::ImplParams params;
  std::vector<int> twists({1,1,1,0});
  params.twists = twists;
    
  GparityMobiusFermionR reg_action(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_b-1.,params);

  XconjugateMobiusFermionR::ImplParams xparams;
  xparams.twists = twists;
  xparams.boundary_phase = 1.0;
  
  XconjugateMobiusFermionR xconj_action(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_b-1.,xparams);

  xparams.boundary_phase = -1.0;
  XconjugateMobiusFermionR xbarconj_action(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_b-1.,xparams);
 

  //#######################################################################################################################################
  {
    std::cout << "Testing M on full grid" << std::endl;

    FermionField1f rand_sc(FGrid), rand_sc2(FGrid);
    gaussian(RNG5,rand_sc);   
    gaussian(RNG5,rand_sc2);

    FermionField2f rand_f0(FGrid); //v \delta_f,0
    rand_f0 = Zero();    
    PokeIndex<GparityFlavourIndex>(rand_f0, rand_sc, 0);

    FermionField2f rand_f1(FGrid); //v \delta_f,1
    rand_f1 = Zero();
    PokeIndex<GparityFlavourIndex>(rand_f1, rand_sc, 1);
    
    FermionField2f tmp(FGrid), tmp2(FGrid), tmp3(FGrid), tmp4(FGrid), tmp5(FGrid);
    FermionField1f tmpsc(FGrid), tmpsc2(FGrid), tmpsc3(FGrid), tmpsc4(FGrid);
    RealD nrm;

    std::cout << "Test the relationship between the upper and lower rows in flavor space of the G-parity Dirac operator" << std::endl;

    //Test  M00 v =  -X M11* X v = -X [M11 X v*]*
    reg_action.M(rand_f0,tmp);
    FermionField1f M00v = PeekIndex<GparityFlavourIndex>(tmp,0);
    FermionField1f M10v = PeekIndex<GparityFlavourIndex>(tmp,1);

    tmp = X*conjugate(rand_f1);    
    reg_action.M(tmp,tmp2);
    FermionField1f M11Xvconj = PeekIndex<GparityFlavourIndex>(tmp2,1);
    FermionField1f M01Xvconj = PeekIndex<GparityFlavourIndex>(tmp2,0);

    tmpsc = -(X*conjugate(M11Xvconj));
    nrm = norm2Diff(tmpsc, M00v);
    std::cout << "Test of M00 v =  -X M11* X v = -X [M11 X v*]* (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    //Test  M10 v =  X M01* X v = X [M01 X v*]*
    tmpsc = X*conjugate(M01Xvconj);
    nrm = norm2Diff(tmpsc, M10v);
    std::cout << "Test of M10 v =  X M11* X v = X [M11 X v*]*  (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    //Test the X-conjugate implementation
    std::cout << "Test the implementation of the X-conjugate action against the reference"<< std::endl;
    XconjugateDWF<GparityMobiusFermionR> xconj_action_ref(&reg_action);
    XconjugateDWF<GparityMobiusFermionR> xbarconj_action_ref(&reg_action,true);

    //Test upper boundary
    std::vector<int> L(4);
    for(int mu=0;mu<4;mu++)
      L[mu] = UGrid->GlobalDimensions()[mu];
    Coordinate site(5,0);
    typedef FermionField1f::vector_object SpinorV;
    typedef typename SpinorV::scalar_object SpinorS;

    tmpsc3 = Zero();
    for(int i=2;i<5;i++) site[i] = L[i-1]/2; //midpoint in y,z,t
    site[1] = 0; //field only on site on lower boundary

    SpinorS v;
    peekSite(v, rand_sc, site);    
    pokeSite(v, tmpsc3, site);

    xconj_action_ref.M(tmpsc3, tmpsc);
    xconj_action.M(tmpsc3, tmpsc2);
    nrm = norm2Diff(tmpsc, tmpsc2);
    std::cout << "Test of Xconjugate action M upper boundary only (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    site[1] = L[0]-1;

    SpinorS r1,r2;
    peekSite(r1, tmpsc, site);
    peekSite(r2, tmpsc2, site);
    nrm = norm2Diff(r1,r2);
    std::cout << "Results L-1\nref:  " << r1 << "\nimpl: " << r2 << "\ndiff (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    site[1] = 1;
    peekSite(r1, tmpsc, site);
    peekSite(r2, tmpsc2, site);
    nrm = norm2Diff(r1,r2);
    std::cout << "Results 1\nref:  " << r1 << "\nimpl: " << r2 << "\ndiff (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    //Test lower boundary
    site[1] = L[0]-1;
    tmpsc3 = Zero();
    pokeSite(v, tmpsc3, site);


    xconj_action_ref.M(tmpsc3, tmpsc);
    xconj_action.M(tmpsc3, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of Xconjugate action M lower boundary only (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    site[1] = 0;

    peekSite(r1, tmpsc, site);
    peekSite(r2, tmpsc2, site);
    nrm = norm2Diff(r1,r2);
    std::cout << "Results 0\nref:  " << r1 << "\nimpl: " << r2 << "\ndiff (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    site[1] = L[0]-2;

    peekSite(r1, tmpsc, site);
    peekSite(r2, tmpsc2, site);
    nrm = norm2Diff(r1,r2);
    std::cout << "Results L-2\nref:  " << r1 << "\nimpl: " << r2 << "\ndiff (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    //Test full
    xconj_action_ref.M(rand_sc, tmpsc);
    xconj_action.M(rand_sc, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of Xconjugate action M against reference, full field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    xconj_action_ref.Mdag(rand_sc, tmpsc);
    xconj_action.Mdag(rand_sc, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of Xconjugate action Mdag against reference, full field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    xbarconj_action_ref.M(rand_sc, tmpsc);
    xbarconj_action.M(rand_sc, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of Xbar-conjugate action M against reference, full field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    xbarconj_action_ref.Mdag(rand_sc, tmpsc);
    xbarconj_action.Mdag(rand_sc, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of Xbar-conjugate action Mdag against reference, full field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    RealD u, l;

    //Test the X-conjugate Dirac op acting on a field is the same as the G-parity Dirac op acting on the equivalent 2f field
    xconj_action.M(rand_sc, tmpsc);
    boostXconjToGparity(tmp, rand_sc);
    reg_action.M(tmp, tmp2);
    norm2DiffXconj(u,l,tmp2,tmpsc);
    std::cout << "Test X-conj Dop reproduces G-parity Dop acting on X-conjugate field, f=0 (expect 0): " << u << " f=1 (expect 0): " << l << std::endl;
    assert(l < 1e-10);
    assert(u < 1e-10);

    //Test the Xbar-conjugate Dirac op acting on a field is the same as the G-parity Dirac op acting on the equivalent 2f field
    xbarconj_action.M(rand_sc, tmpsc);
    boostXbarConjToGparity(tmp, rand_sc);
    reg_action.M(tmp, tmp2);
    norm2DiffXbarConj(u,l,tmp2,tmpsc);
    std::cout << "Test Xbar-conj Dop reproduces G-parity Dop acting on Xbar-conjugate field, f=0 (expect 0): " << u << " f=1 (expect 0): " << l << std::endl;
    assert(l < 1e-10);
    assert(u < 1e-10);

    //Test the X-conj 2f wrapper reproduces G-parity Dop acting on X-conjugate field 
    Xconjugate2fWrapper<XconjugateMobiusFermionR, GparityMobiusFermionR> xconj_2f_wrapper(&xconj_action,&reg_action);
    boostXconjToGparity(tmp, rand_sc);
    xconj_2f_wrapper.M(tmp, tmp3);
    reg_action.M(tmp, tmp2);
    nrm = norm2Diff(tmp3,tmp2);
    std::cout << "Test the X-conj 2f wrapper reproduces G-parity Dop acting on X-conjugate field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);
  }

  //########################################################################
  {
    std::cout << "Test of red-black preconditioned operator" << std::endl;
    SchurDiagMooeeOperator<GparityMobiusFermionR,FermionField2f> reg_schurop(reg_action);
    SchurDiagMooeeOperator<XconjugateMobiusFermionR,FermionField1f> xconj_schurop(xconj_action);
    SchurDiagMooeeOperator<XconjugateMobiusFermionR,FermionField1f> xbarconj_schurop(xbarconj_action);
    XconjugateDWF<GparityMobiusFermionR> xconj_action_ref(&reg_action);

    FermionField1f rand_sc(FrbGrid), rand_sc2(FrbGrid); //v
    gaussian(RNG5rb,rand_sc);
    gaussian(RNG5rb,rand_sc2);

    FermionField2f rand_f0(FrbGrid); //v \delta_f,0
    rand_f0 = Zero();    
    PokeIndex<GparityFlavourIndex>(rand_f0, rand_sc, 0);

    FermionField2f rand_f1(FrbGrid); //v \delta_f,1
    rand_f1 = Zero();
    PokeIndex<GparityFlavourIndex>(rand_f1, rand_sc, 1);
    
    FermionField2f tmp(FrbGrid), tmp2(FrbGrid), tmp3(FrbGrid), tmp4(FrbGrid);
    FermionField1f tmpsc(FrbGrid), tmpsc2(FrbGrid), tmpsc3(FrbGrid);
    RealD nrm;

    std::cout << "Test the relationship between the upper and lower rows in flavor space of the G-parity preconditioned Dirac operator" << std::endl;    
    
    reg_schurop.Mpc(rand_f0,tmp);
    FermionField1f M00v = PeekIndex<GparityFlavourIndex>(tmp,0);
    FermionField1f M10v = PeekIndex<GparityFlavourIndex>(tmp,1);

    //Test  M00 v =  -X M11* X v = -X [M11 X v*]*
    tmp = X*conjugate(rand_f1);    

    reg_schurop.Mpc(tmp,tmp2);
    FermionField1f M11Xvconj = PeekIndex<GparityFlavourIndex>(tmp2,1);
    FermionField1f M01Xvconj = PeekIndex<GparityFlavourIndex>(tmp2,0);

    tmpsc = -(X*conjugate(M11Xvconj));
    nrm = norm2Diff(tmpsc, M00v);
    std::cout << "Test of M00 v =  -X M11* X v = -X [M11 X v*]*  (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    //Test  M10 v =  X M01* X v = X [M01 X v*]*
    tmpsc = X*conjugate(M01Xvconj);
    nrm = norm2Diff(tmpsc,M10v);
    std::cout << "Test of M10 v =  X M11* X v = X [M11 X v*]*  (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    std::cout << "Test the relationship between the upper and lower rows in flavor space of the G-parity preconditioned squared Dirac operator" << std::endl;    

    reg_schurop.HermOp(rand_f0,tmp);
    M00v = PeekIndex<GparityFlavourIndex>(tmp,0);
    M10v = PeekIndex<GparityFlavourIndex>(tmp,1);

    //Test  M00 v =  -X M11* X v = -X [M11 X v*]*
    tmp = X*conjugate(rand_f1);    

    reg_schurop.HermOp(tmp,tmp2);
    M11Xvconj = PeekIndex<GparityFlavourIndex>(tmp2,1);
    M01Xvconj = PeekIndex<GparityFlavourIndex>(tmp2,0);

    tmpsc = -(X*conjugate(M11Xvconj));
    nrm = norm2Diff(tmpsc,M00v);
    std::cout << "Test of M00 v =  -X M11* X v = -X [M11 X v*]*  (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    //Test  M10 v =  X M01* X v = X [M01 X v*]*
    tmpsc = X*conjugate(M01Xvconj);
    nrm = norm2Diff(tmpsc,M10v);
    std::cout << "Test of M10 v =  X M11* X v = X [M11 X v*]*  (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    //Test the X-conjugate implementation
    std::cout << "Test the implementation of the X-conjugate preconditioned action against the reference"<< std::endl;
    xconj_action_ref.Meooe(rand_sc, tmpsc);
    xconj_action.Meooe(rand_sc, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of X-conjugate action Meooe against reference, full field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    xconj_action_ref.MeooeDag(rand_sc, tmpsc);
    xconj_action.MeooeDag(rand_sc, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of X-conjugate action MeooeDag against reference, full field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    xconj_action_ref.Mooee(rand_sc, tmpsc);
    xconj_action.Mooee(rand_sc, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of X-conjugate action Mooee against reference, full field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    xconj_action_ref.MooeeDag(rand_sc, tmpsc);
    xconj_action.MooeeDag(rand_sc, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of X-conjugate action MooeeDag against reference, full field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);
        
    xconj_action_ref.MooeeInv(rand_sc, tmpsc);
    xconj_action.MooeeInv(rand_sc, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of X-conjugate action MooeeInv against reference, full field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    xconj_action_ref.MooeeInvDag(rand_sc, tmpsc);
    xconj_action.MooeeInvDag(rand_sc, tmpsc2);
    nrm = norm2Diff(tmpsc,tmpsc2);
    std::cout << "Test of X-conjugate action MooeeInvDag against reference, full field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    RealD u,l;

    //Test the X-conjugate Dirac op acting on a field is the same as the G-parity Dirac op acting on a field with explicit X-conjugate BCs
    xconj_schurop.HermOp(rand_sc, tmpsc);
    boostXconjToGparity(tmp, rand_sc);
    reg_schurop.HermOp(tmp, tmp2);
    norm2DiffXconj(u,l,tmp2,tmpsc);
    std::cout << "Test X-conj HermOp reproduces G-parity HermOp acting on X-conjugate field, f=0 (expect 0): " << u << " f=1 (expect 0): " << l << std::endl;
    assert(u < 1e-10);
    assert(l < 1e-10);
    
    //Test the X-conj 2f wrapper reproduces G-parity Dop acting on X-conjugate field
    Xconjugate2fWrapper<XconjugateMobiusFermionR,GparityMobiusFermionR> xconj_2f_wrapper(&xconj_action,&reg_action);
    SchurDiagMooeeOperator<Xconjugate2fWrapper<XconjugateMobiusFermionR,GparityMobiusFermionR>, FermionField2f> xconj_2f_schurop(xconj_2f_wrapper);
    boostXconjToGparity(tmp, rand_sc);
    reg_schurop.HermOp(tmp, tmp2);
    xconj_2f_schurop.HermOp(tmp,tmp3);
    nrm = norm2Diff(tmp3,tmp2);
    std::cout << "Test the X-conj 2f wrapper reproduces G-parity HermOp acting on X-conjugate field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    //Test the Xbar-conjugate Dirac op acting on a field is the same as the G-parity Dirac op acting on a field with explicit Xbar-conjugate BCs
    xbarconj_schurop.HermOp(rand_sc, tmpsc);
    boostXbarConjToGparity(tmp,rand_sc);
    reg_schurop.HermOp(tmp, tmp2);
    norm2DiffXbarConj(u,l,tmp2,tmpsc);
    std::cout << "Test Xbar-conj HermOp reproduces G-parity HermOp acting on Xbar-conjugate field, f=0 (expect 0): " << u << " f=1 (expect 0): " << l << std::endl;
    assert(u < 1e-10);
    assert(l < 1e-10);   

    //Test the Xbar-conj 2f wrapper reproduces G-parity Dop acting on Xbar-conjugate field
    Xconjugate2fWrapper<XconjugateMobiusFermionR,GparityMobiusFermionR> xbarconj_2f_wrapper(&xbarconj_action,&reg_action);
    SchurDiagMooeeOperator<Xconjugate2fWrapper<XconjugateMobiusFermionR,GparityMobiusFermionR>, FermionField2f> xbarconj_2f_schurop(xbarconj_2f_wrapper);
    boostXbarConjToGparity(tmp, rand_sc);
    reg_schurop.HermOp(tmp, tmp2);
    xbarconj_2f_schurop.HermOp(tmp,tmp3);
    nrm = norm2Diff(tmp3,tmp2);
    std::cout << "Test the Xbar-conj 2f wrapper reproduces G-parity HermOp acting on Xbar-conjugate field (expect 0): " << nrm << std::endl;
    assert(nrm < 1e-10);

    //Test reconstruction of G-parity Dop on arbitrary flavor vector using Xconj action
    PokeIndex<GparityFlavourIndex>(tmp, rand_sc, 0);
    PokeIndex<GparityFlavourIndex>(tmp, rand_sc2, 1);
    reg_schurop.HermOp(tmp, tmp2);
    
    FermionField1f rho(FrbGrid), xi(FrbGrid);
    rho = 0.5 * ( rand_sc + (X*conjugate(rand_sc2)) );
    xi = 0.5 * ( rand_sc - (X*conjugate(rand_sc2)) );
    xconj_schurop.HermOp(rho, tmpsc);
    xbarconj_schurop.HermOp(xi, tmpsc2);

    tmpsc3 = PeekIndex<GparityFlavourIndex>(tmp2,0) - tmpsc - tmpsc2;
    u = norm2(tmpsc3);

    tmpsc = -(X*conjugate(tmpsc));
    tmpsc2 = X*conjugate(tmpsc2);
    tmpsc3 = PeekIndex<GparityFlavourIndex>(tmp2,1) - tmpsc - tmpsc2;
    l = norm2(tmpsc3);

    std::cout << "Test reconstruction of GP HermOp on random 2f field from Xconj ops f=0 (expect 0): " << u << " f=1 (expect 0): " << l << std::endl;
    assert(u < 1e-10);
    assert(l < 1e-10);
  }
  std::cout << "All tests passed" << std::endl;
  
  Grid_finalize();
}

