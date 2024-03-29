/*************************************************************************************

    grid` physics library, www.github.com/paboyle/Grid 

    Copyright (C) 2015

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
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
//#include <Grid/Grid.h>

#ifndef GRID_QCD_GAUGE_FIX_H
#define GRID_QCD_GAUGE_FIX_H

NAMESPACE_BEGIN(Grid);


template <class Gimpl> 
class FourierAcceleratedGaugeFixer  : public Gimpl {
public:
  INHERIT_GIMPL_TYPES(Gimpl);

  typedef typename Gimpl::GaugeLinkField GaugeMat;
  typedef typename Gimpl::GaugeField GaugeLorentz;

  //A_\mu(x) = -i Ta(U_\mu(x) )   where Ta(U) = 1/2( U - U^dag ) - 1/2N tr(U - U^dag)  is the traceless antihermitian part. This is an O(A^3) approximation to the logarithm of U
  static void GaugeLinkToLieAlgebraField(const GaugeMat &U, GaugeMat &A) {
    Complex cmi(0.0,-1.0);
    A = Ta(U) * cmi;
  }
  
  //The derivative of the Lie algebra field
  static void DmuAmu(const std::vector<GaugeMat> &U, GaugeMat &dmuAmu,int orthog) {
    GridBase* grid = U[0].Grid();
    GaugeMat Ax(grid);
    GaugeMat Axm1(grid);
    GaugeMat Utmp(grid);

    dmuAmu=Zero();
    for(int mu=0;mu<Nd;mu++){
      if ( mu != orthog ) {
	//Rather than define functionality to work out how the BCs apply to A_\mu we simply use the BC-aware Cshift to the gauge links and compute A_\mu(x) and A_\mu(x-1) separately
	//Ax = A_\mu(x)
	GaugeLinkToLieAlgebraField(U[mu], Ax);
	
	//Axm1 = A_\mu(x_\mu-1)
	Utmp = Gimpl::CshiftLink(U[mu], mu, -1);
	GaugeLinkToLieAlgebraField(Utmp, Axm1);
	
	//Derivative
	dmuAmu = dmuAmu + Ax - Axm1;
      }
    }
  }  

  //Fix the gauge field Umu
  //0 < alpha < 1 is related to the step size, cf https://arxiv.org/pdf/1405.5812.pdf
  static void SteepestDescentGaugeFix(GaugeLorentz &Umu,Real alpha,int maxiter,Real Omega_tol, Real Phi_tol,bool Fourier=false,int orthog=-1,bool err_on_no_converge=true) {
    GridBase *grid = Umu.Grid();
    GaugeMat xform(grid);
    SteepestDescentGaugeFix(Umu,xform,alpha,maxiter,Omega_tol,Phi_tol,Fourier,orthog,err_on_no_converge);
  }
  static void SteepestDescentGaugeFix(GaugeLorentz &Umu,GaugeMat &xform,Real alpha,int maxiter,Real Omega_tol, Real Phi_tol,bool Fourier=false,int orthog=-1,bool err_on_no_converge=true) {
  //Fix the gauge field Umu and also return the gauge transformation from the original gauge field, xform

    GridBase *grid = Umu.Grid();

    Real org_plaq      =WilsonLoops<Gimpl>::avgPlaquette(Umu);
    Real org_link_trace=WilsonLoops<Gimpl>::linkTrace(Umu); 
    Real old_trace = org_link_trace;
    Real trG;
    
    xform=1.0;

    std::vector<GaugeMat> U(Nd,grid);
    GaugeMat dmuAmu(grid);

    {
      Real plaq      =WilsonLoops<Gimpl>::avgPlaquette(Umu);
      Real link_trace=WilsonLoops<Gimpl>::linkTrace(Umu); 
      if( (orthog>=0) && (orthog<Nd) ){
	std::cout << GridLogMessage << " Gauge fixing to Coulomb gauge time="<<orthog<< " plaq= "<<plaq<<" link trace = "<<link_trace<<  std::endl;
      } else { 
	std::cout << GridLogMessage << " Gauge fixing to Landau gauge plaq= "<<plaq<<" link trace = "<<link_trace<<  std::endl;
      }
    }
    for(int i=0;i<maxiter;i++){

      for(int mu=0;mu<Nd;mu++) U[mu]= PeekIndex<LorentzIndex>(Umu,mu);

      trG = GaugeFixStep(U,xform,alpha,dmuAmu,orthog,Fourier);

      //      std::cout << GridLogMessage << "trG   "<< trG<< std::endl;
      //      std::cout << GridLogMessage << "xform "<< norm2(xform)<< std::endl;
      //      std::cout << GridLogMessage << "dmuAmu "<< norm2(dmuAmu)<< std::endl;

      for(int mu=0;mu<Nd;mu++) PokeIndex<LorentzIndex>(Umu,U[mu],mu);
      // Monitor progress and convergence test 
      // infrequently to minimise cost overhead
      if ( i %20 == 0 ) { 
	Real plaq      =WilsonLoops<Gimpl>::avgPlaquette(Umu);
	Real link_trace=WilsonLoops<Gimpl>::linkTrace(Umu); 

	if (Fourier) 
	  std::cout << GridLogMessage << "Fourier Iteration "<<i<< " plaq= "<<plaq<< " dmuAmu " << norm2(dmuAmu)<< std::endl;
	else 
	  std::cout << GridLogMessage << " Iteration "<<i<< " plaq= "<<plaq<< " dmuAmu " << norm2(dmuAmu)<< std::endl;
	
	Real Phi  = 1.0 - old_trace / link_trace ;
	Real Omega= 1.0 - trG;

	std::cout << GridLogMessage << " Iteration "<<i<< " Phi= "<<Phi<< " Omega= " << Omega<< " trG " << trG <<std::endl;
	if ( (Omega < Omega_tol) && ( ::fabs(Phi) < Phi_tol) ) {
	  std::cout << GridLogMessage << "Converged ! "<<std::endl;
	  return;
	}

	old_trace = link_trace;

      }
    }
    std::cout << GridLogError << "Gauge fixing did not converge in " << maxiter << " iterations." << std::endl;
    if (err_on_no_converge)
      assert(0 && "Gauge fixing did not converge within the specified number of iterations");
  };

  static Real GaugeFixStep(std::vector<GaugeMat> &U,GaugeMat &xform, Real alpha, GaugeMat & dmuAmu,int orthog,bool Fourier) {

    GridBase *grid = U[0].Grid();

    Real vol = grid->gSites();

    FFT theFFT((GridCartesian *)grid);

    GaugeMat g(grid);
    GaugeMat dmuAmu_p(grid);
    DmuAmu(U,dmuAmu,orthog);

    //The eigenmodes of the Laplacian are boundary-condition dependent
    if(Fourier) Gimpl::FourierAcceleratedGfix(dmuAmu, orthog);

    GaugeMat ciadmam(grid);
    Complex cialpha(0.0,-alpha);
    ciadmam = dmuAmu*cialpha;
    SU<Nc>::taExp(ciadmam,g);

    Real trG = TensorRemove(sum(trace(g))).real()/vol/Nc;

    xform = g*xform ;
    SU<Nc>::GaugeTransform<Gimpl>(U,g);

    return trG;
  }

  static void ExpiAlphaDmuAmu(const std::vector<GaugeMat> &U,GaugeMat &g, Real alpha, GaugeMat &dmuAmu,int orthog) {
    GridBase *grid = g.Grid();
    Complex cialpha(0.0,-alpha);
    GaugeMat ciadmam(grid);
    DmuAmu(U,dmuAmu,orthog);
    ciadmam = dmuAmu*cialpha;
    SU<Nc>::taExp(ciadmam,g);
  }  
};

NAMESPACE_END(Grid);

#endif
