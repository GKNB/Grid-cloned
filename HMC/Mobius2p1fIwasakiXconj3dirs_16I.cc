/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./HMC/Mobius2p1fIDSDRGparityEOFA.cc

Copyright (C) 2015-2016

Author: Christopher Kelly <ckelly@bnl.gov>
Author: Peter Boyle <pabobyle@ph.ed.ac.uk>


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
#include <Grid/Grid.h>

using namespace Grid;

//Production binary for the 40ID G-parity ensemble

struct RatQuoParameters: Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(RatQuoParameters,
				  double, bnd_lo,
				  double, bnd_hi,
				  Integer, action_degree,
				  double, action_tolerance,
				  Integer, md_degree,
				  double, md_tolerance,
				  Integer, reliable_update_freq,
				  Integer, bnd_check_freq);
  RatQuoParameters() { 
    bnd_lo = 1e-2;
    bnd_hi = 30;
    action_degree = 10;
    action_tolerance = 1e-10;
    md_degree = 10;
    md_tolerance = 1e-8;
    bnd_check_freq = 20;
    reliable_update_freq = 50;
  }

  void Export(RationalActionParams &into) const{
    into.lo = bnd_lo;
    into.hi = bnd_hi;
    into.action_degree = action_degree;
    into.action_tolerance = action_tolerance;
    into.md_degree = md_degree;
    into.md_tolerance = md_tolerance;
    into.BoundsCheckFreq = bnd_check_freq;
  }
};

struct EOFAparameters: Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(EOFAparameters,
				  OneFlavourRationalParams, rat_params,
				  double, action_tolerance,
				  double, action_mixcg_inner_tolerance,
				  double, md_tolerance,
				  double, md_mixcg_inner_tolerance);

  EOFAparameters() { 
    action_mixcg_inner_tolerance = 1e-8;
    action_tolerance = 1e-10;
    md_tolerance = 1e-8;
    md_mixcg_inner_tolerance = 1e-8;

    rat_params.lo = 1.0;
    rat_params.hi = 25.0;
    rat_params.MaxIter  = 50000;
    rat_params.tolerance= 1.0e-9;
    rat_params.degree   = 14;
    rat_params.precision= 50;
  }
};


struct TwoFlavorActionParameters: Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(TwoFlavorActionParameters,
				  double, action_tolerance,
				  double, md_tolerance);

  TwoFlavorActionParameters() { 
    action_tolerance = 1e-10;
    md_tolerance = 1e-8;
  }
};


struct EvolParameters: Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(EvolParameters,
                                  Integer, StartTrajectory,
                                  Integer, Trajectories,
				  Integer, SaveInterval,
				  Integer, Steps,
				  RealD, TrajectoryLength,
                                  bool, MetropolisTest,
				  std::string, StartingType,
				  std::vector<Integer>, GparityDirs,
				  std::vector<TwoFlavorActionParameters>, ratio_l,
				  EOFAparameters, eofa_s);

  EvolParameters() {
    //For initial thermalization; afterwards user should switch Metropolis on and use StartingType=CheckpointStart
    MetropolisTest    = false;
    StartTrajectory   = 0;
    Trajectories      = 50;
    SaveInterval = 5;
    StartingType      = "ColdStart";
    GparityDirs.resize(3, 1); //1 for G-parity, 0 for periodic
    Steps = 5;
    TrajectoryLength = 1.0;
  }
};

bool fileExists(const std::string &fn){
  std::ifstream f(fn);
  return f.good();
}




struct LanczosParameters: Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(LanczosParameters,
				  double, alpha,
				  double, beta,
				  double, mu,
				  int, ord,
				  int, n_stop,
				  int, n_want,
				  int, n_use,
				  double, tolerance);

  LanczosParameters() {
    alpha = 35;
    beta = 5;
    mu = 0;
    ord = 100;
    n_stop = 10;
    n_want = 10;
    n_use = 15;
    tolerance = 1e-6;
  }
};



template<typename FermionActionD, typename FermionFieldD>
void computeEigenvalues(std::string param_file,
			GridCartesian* Grid, GridRedBlackCartesian* rbGrid, const LatticeGaugeFieldD &latt,  //expect lattice to have been initialized to something
			FermionActionD &action, GridParallelRNG &rng){
  
  LanczosParameters params;
  if(fileExists(param_file)){
    std::cout << GridLogMessage << " Reading " << param_file << std::endl;
    Grid::XmlReader rd(param_file);
    read(rd, "LanczosParameters", params);
  }else if(!GlobalSharedMemory::WorldRank){
    std::cout << GridLogMessage << " File " << param_file << " does not exist" << std::endl;
    std::cout << GridLogMessage << " Writing xml template to " << param_file << ".templ" << std::endl;
    Grid::XmlWriter wr(param_file + ".templ");
    write(wr, "LanczosParameters", params);
  }

  FermionFieldD gauss_o(rbGrid);
  FermionFieldD gauss(Grid);
  gaussian(rng, gauss);
  pickCheckerboard(Odd, gauss_o, gauss);

  action.ImportGauge(latt);

  SchurDiagMooeeOperator<FermionActionD, FermionFieldD> hermop(action);
  PlainHermOp<FermionFieldD> hermop_wrap(hermop);
  //ChebyshevLanczos<FermionFieldD> Cheb(params.alpha, params.beta, params.mu, params.ord);
  assert(params.mu == 0.0);

  Chebyshev<FermionFieldD> Cheb(params.beta*params.beta, params.alpha*params.alpha, params.ord+1);
  FunctionHermOp<FermionFieldD> Cheb_wrap(Cheb, hermop);

  std::cout << "IRL: alpha=" << params.alpha << " beta=" << params.beta << " mu=" << params.mu << " ord=" << params.ord << std::endl;
  ImplicitlyRestartedLanczos<FermionFieldD> IRL(Cheb_wrap, hermop_wrap, params.n_stop, params.n_want, params.n_use, params.tolerance, 50000);

  std::vector<RealD> eval(params.n_use);
  std::vector<FermionFieldD> evec(params.n_use, rbGrid);
  int Nconv;
  IRL.calc(eval, evec, gauss_o, Nconv);

  std::cout << "Eigenvalues:" << std::endl;
  for(int i=0;i<params.n_want;i++){
    std::cout << i << " " << eval[i] << std::endl;
  }
}


//Check the quality of the RHMC approx
//action_or_md toggles checking the action (0), MD (1) or both (2) setups
template<typename FermionActionD, typename FermionFieldD, typename RHMCtype>
void checkRHMC(GridCartesian* Grid, GridRedBlackCartesian* rbGrid, const LatticeGaugeFieldD &latt,  //expect lattice to have been initialized to something
	       FermionActionD &numOp, FermionActionD &denOp, RHMCtype &rhmc, GridParallelRNG &rng,
	       int inv_pow, const std::string &quark_descr, int action_or_md){
  assert(action_or_md == 0 || action_or_md == 1 || action_or_md == 2);
  
  FermionFieldD gauss_o(rbGrid);
  FermionFieldD gauss(Grid);
  gaussian(rng, gauss);
  pickCheckerboard(Odd, gauss_o, gauss);

  numOp.ImportGauge(latt);
  denOp.ImportGauge(latt);

  typedef typename FermionActionD::Impl_t FermionImplPolicyD;
  SchurDifferentiableOperator<FermionImplPolicyD> MdagM(numOp);
  SchurDifferentiableOperator<FermionImplPolicyD> VdagV(denOp);

  PowerMethod<FermionFieldD> power_method;
  RealD lambda_max;

  std::cout << "Starting: Get RHMC high bound approx for " << quark_descr << " numerator" << std::endl;

  lambda_max = power_method(MdagM,gauss_o);
  std::cout << GridLogMessage << "Got lambda_max "<<lambda_max<<std::endl;

  std::cout << "Starting: Get RHMC high bound approx for " << quark_descr << " denominator" << std::endl;
  lambda_max = power_method(VdagV,gauss_o);
  std::cout << GridLogMessage << "Got lambda_max "<<lambda_max<<std::endl;

  if(action_or_md == 0 || action_or_md == 2){
    std::cout << "Starting: Checking quality of RHMC action approx for " << quark_descr << " quark numerator and power -1/" << inv_pow << std::endl;
    InversePowerBoundsCheck(inv_pow, 50000, 1e16, MdagM,gauss_o, rhmc.ApproxNegPowerAction); //use large tolerance to prevent exit on fail; we are trying to tune here!
    std::cout << "Finished: Checking quality of RHMC action approx for " << quark_descr << " quark numerator and power -1/" << inv_pow << std::endl;

    std::cout << "Starting: Checking quality of RHMC action approx for " << quark_descr << " quark numerator and power -1/" << 2*inv_pow << std::endl;
    InversePowerBoundsCheck(2*inv_pow, 50000, 1e16, MdagM,gauss_o, rhmc.ApproxNegHalfPowerAction);
    std::cout << "Finished: Checking quality of RHMC action approx for " << quark_descr << " quark numerator and power -1/" << 2*inv_pow << std::endl;

    std::cout << "Starting: Checking quality of RHMC action approx for " << quark_descr << " quark denominator and power -1/" << inv_pow << std::endl;
    InversePowerBoundsCheck(inv_pow, 50000, 1e16, VdagV,gauss_o, rhmc.ApproxNegPowerAction);
    std::cout << "Finished: Checking quality of RHMC action approx for " << quark_descr << " quark denominator and power -1/" << inv_pow << std::endl;

    std::cout << "Starting: Checking quality of RHMC action approx for " << quark_descr << " quark denominator and power -1/" << 2*inv_pow << std::endl;
    InversePowerBoundsCheck(2*inv_pow, 50000, 1e16, VdagV,gauss_o, rhmc.ApproxNegHalfPowerAction);
    std::cout << "Finished: Checking quality of RHMC action approx for " << quark_descr << " quark denominator and power -1/" << 2*inv_pow << std::endl;
  }

  std::cout << "-------------------------------------------------------------------------------" << std::endl;

  if(action_or_md == 1 || action_or_md == 2){
    std::cout << "Starting: Checking quality of RHMC MD approx for " << quark_descr << " quark numerator and power -1/" << inv_pow << std::endl;
    InversePowerBoundsCheck(inv_pow, 50000, 1e16, MdagM,gauss_o, rhmc.ApproxNegPowerMD); 
    std::cout << "Finished: Checking quality of RHMC MD approx for " << quark_descr << " quark numerator and power -1/" << inv_pow << std::endl;

    std::cout << "Starting: Checking quality of RHMC MD approx for " << quark_descr << " quark numerator and power -1/" << 2*inv_pow << std::endl;
    InversePowerBoundsCheck(2*inv_pow, 50000, 1e16, MdagM,gauss_o, rhmc.ApproxNegHalfPowerMD);
    std::cout << "Finished: Checking quality of RHMC MD approx for " << quark_descr << " quark numerator and power -1/" << 2*inv_pow << std::endl;

    std::cout << "Starting: Checking quality of RHMC MD approx for " << quark_descr << " quark denominator and power -1/" << inv_pow << std::endl;
    InversePowerBoundsCheck(inv_pow, 50000, 1e16, VdagV,gauss_o, rhmc.ApproxNegPowerMD);
    std::cout << "Finished: Checking quality of RHMC MD approx for " << quark_descr << " quark denominator and power -1/" << inv_pow << std::endl;

    std::cout << "Starting: Checking quality of RHMC MD approx for " << quark_descr << " quark denominator and power -1/" << 2*inv_pow << std::endl;
    InversePowerBoundsCheck(2*inv_pow, 50000, 1e16, VdagV,gauss_o, rhmc.ApproxNegHalfPowerMD);
    std::cout << "Finished: Checking quality of RHMC MD approx for " << quark_descr << " quark denominator and power -1/" << 2*inv_pow << std::endl;
  }
}


template<typename FermionImplPolicy>
void checkEOFA(ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> &EOFA,
	       GridCartesian* FGrid, GridParallelRNG &rng, const LatticeGaugeFieldD &latt){
  std::cout << GridLogMessage << "Starting EOFA action/bounds check" << std::endl;
  typename FermionImplPolicy::FermionField eta(FGrid);
  RealD scale = std::sqrt(0.5);
  gaussian(rng,eta); eta = eta * scale;

  //Use the inbuilt check
  EOFA.refresh(latt, eta);
  EOFA.S(latt);
  std::cout << GridLogMessage << "Finished EOFA upper action/bounds check" << std::endl;
}


template<typename FermionImplPolicy>
class EOFAlinop: public LinearOperatorBase<typename FermionImplPolicy::FermionField>{
  ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> &EOFA;
  LatticeGaugeFieldD &U;
public:
  EOFAlinop(ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> &EOFA, LatticeGaugeFieldD &U): EOFA(EOFA), U(U){}

  typedef typename FermionImplPolicy::FermionField Field;
  void OpDiag (const Field &in, Field &out){ assert(0); }
  void OpDir  (const Field &in, Field &out,int dir,int disp){ assert(0); }
  void OpDirAll  (const Field &in, std::vector<Field> &out){ assert(0); } 

  void Op     (const Field &in, Field &out){ assert(0); }
  void AdjOp  (const Field &in, Field &out){ assert(0); }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){ assert(0); }
  void HermOp(const Field &in, Field &out){ EOFA.Meofa(U, in, out); }
};

template<typename FermionImplPolicy>
void upperBoundEOFA(ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> &EOFA,
		    GridCartesian* FGrid, GridParallelRNG &rng, LatticeGaugeFieldD &latt){
  std::cout << GridLogMessage << "Starting EOFA upper bound compute" << std::endl;
  EOFAlinop<FermionImplPolicy> linop(EOFA, latt);
  typename FermionImplPolicy::FermionField eta(FGrid);
  gaussian(rng,eta);
  PowerMethod<typename FermionImplPolicy::FermionField> power_method(1e-8,1000); //just let it run until we get bored
  auto lambda_max = power_method(linop,eta);
  std::cout << GridLogMessage << "Upper bound of EOFA operator " << lambda_max << std::endl;
}

//Applications of M^{-1} cost the same as M for EOFA!
template<typename FermionImplPolicy>
class EOFAinvLinop: public LinearOperatorBase<typename FermionImplPolicy::FermionField>{
  ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> &EOFA;
  LatticeGaugeFieldD &U;
public:
  EOFAinvLinop(ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> &EOFA, LatticeGaugeFieldD &U): EOFA(EOFA), U(U){}

  typedef typename FermionImplPolicy::FermionField Field;
  void OpDiag (const Field &in, Field &out){ assert(0); }
  void OpDir  (const Field &in, Field &out,int dir,int disp){ assert(0); }
  void OpDirAll  (const Field &in, std::vector<Field> &out){ assert(0); } 

  void Op     (const Field &in, Field &out){ assert(0); }
  void AdjOp  (const Field &in, Field &out){ assert(0); }
  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){ assert(0); }
  void HermOp(const Field &in, Field &out){ EOFA.MeofaInv(U, in, out); }
};

template<typename FermionImplPolicy>
void lowerBoundEOFA(ExactOneFlavourRatioPseudoFermionAction<FermionImplPolicy> &EOFA,
		    GridCartesian* FGrid, GridParallelRNG &rng, LatticeGaugeFieldD &latt){
  std::cout << GridLogMessage << "Starting EOFA lower bound compute using power method on M^{-1}. Inverse of highest eigenvalue is the lowest eigenvalue of M" << std::endl;
  EOFAinvLinop<FermionImplPolicy> linop(EOFA, latt);
  typename FermionImplPolicy::FermionField eta(FGrid);
  gaussian(rng,eta);
  PowerMethod<typename FermionImplPolicy::FermionField> power_method(1e-8,1000); //just let it run until we get bored
  auto lambda_max = power_method(linop,eta);
  std::cout << GridLogMessage << "Lower bound of EOFA operator " << 1./lambda_max << std::endl;
}


NAMESPACE_BEGIN(Grid);

  template<class FermionOperatorD, class FermionOperatorF, class SchurOperatorD, class  SchurOperatorF> 
  class MixedPrecisionConjugateGradientOperatorFunction : public OperatorFunction<typename FermionOperatorD::FermionField> {
  public:
    typedef typename FermionOperatorD::FermionField FieldD;
    typedef typename FermionOperatorF::FermionField FieldF;

    using OperatorFunction<FieldD>::operator();

    RealD   Tolerance;
    RealD   InnerTolerance; //Initial tolerance for inner CG. Defaults to Tolerance but can be changed
    Integer MaxInnerIterations;
    Integer MaxOuterIterations;
    GridBase* SinglePrecGrid4; //Grid for single-precision fields
    GridBase* SinglePrecGrid5; //Grid for single-precision fields
    RealD OuterLoopNormMult; //Stop the outer loop and move to a final double prec solve when the residual is OuterLoopNormMult * Tolerance

    FermionOperatorF &FermOpF;
    FermionOperatorD &FermOpD;;
    SchurOperatorF &LinOpF;
    SchurOperatorD &LinOpD;

    Integer TotalInnerIterations; //Number of inner CG iterations
    Integer TotalOuterIterations; //Number of restarts
    Integer TotalFinalStepIterations; //Number of CG iterations in final patch-up step

    MixedPrecisionConjugateGradientOperatorFunction(RealD tol, 
						    Integer maxinnerit, 
						    Integer maxouterit, 
						    GridBase* _sp_grid4, 
						    GridBase* _sp_grid5, 
						    FermionOperatorF &_FermOpF,
						    FermionOperatorD &_FermOpD,
						    SchurOperatorF   &_LinOpF,
						    SchurOperatorD   &_LinOpD): 
      LinOpF(_LinOpF),
      LinOpD(_LinOpD),
      FermOpF(_FermOpF),
      FermOpD(_FermOpD),
      Tolerance(tol), 
      InnerTolerance(tol), 
      MaxInnerIterations(maxinnerit), 
      MaxOuterIterations(maxouterit), 
      SinglePrecGrid4(_sp_grid4),
      SinglePrecGrid5(_sp_grid5),
      OuterLoopNormMult(100.) 
    { 
    };

    void operator()(LinearOperatorBase<FieldD> &LinOpU, const FieldD &src, FieldD &psi) {

      std::cout << GridLogMessage << " Mixed precision CG wrapper operator() "<<std::endl;

      SchurOperatorD * SchurOpU = static_cast<SchurOperatorD *>(&LinOpU);
      assert(&(SchurOpU->_Mat)==&(LinOpD._Mat));

      precisionChange(FermOpF.Umu, FermOpD.Umu);

      pickCheckerboard(Even,FermOpF.UmuEven,FermOpF.Umu);
      pickCheckerboard(Odd ,FermOpF.UmuOdd ,FermOpF.Umu);

      ////////////////////////////////////////////////////////////////////////////////////
      // Make a mixed precision conjugate gradient
      ////////////////////////////////////////////////////////////////////////////////////
      MixedPrecisionConjugateGradient<FieldD,FieldF> MPCG(Tolerance,MaxInnerIterations,MaxOuterIterations,SinglePrecGrid5,LinOpF,LinOpD);
      MPCG.InnerTolerance = InnerTolerance;
      std::cout << GridLogMessage << "Calling mixed precision Conjugate Gradient" <<std::endl;
      MPCG(src,psi);
    }
  };



  template<class FermionOperatorD, class FermionOperatorF, class SchurOperatorD, class  SchurOperatorF> 
  class MixedPrecisionReliableUpdateConjugateGradientOperatorFunction : public OperatorFunction<typename FermionOperatorD::FermionField> {
  public:
    typedef typename FermionOperatorD::FermionField FieldD;
    typedef typename FermionOperatorF::FermionField FieldF;

    using OperatorFunction<FieldD>::operator();

    RealD Tolerance;
    Integer MaxIterations;

    RealD Delta; //reliable update parameter

    GridBase* SinglePrecGrid4; //Grid for single-precision fields
    GridBase* SinglePrecGrid5; //Grid for single-precision fields

    FermionOperatorF &FermOpF;
    FermionOperatorD &FermOpD;;
    SchurOperatorF &LinOpF;
    SchurOperatorD &LinOpD;
    
    MixedPrecisionReliableUpdateConjugateGradientOperatorFunction(RealD tol, 
								  RealD delta,
								  Integer maxit, 
								  GridBase* _sp_grid4, 
								  GridBase* _sp_grid5, 
								  FermionOperatorF &_FermOpF,
								  FermionOperatorD &_FermOpD,
								  SchurOperatorF   &_LinOpF,
								  SchurOperatorD   &_LinOpD): 
      LinOpF(_LinOpF),
      LinOpD(_LinOpD),
      FermOpF(_FermOpF),
      FermOpD(_FermOpD),
      Tolerance(tol), 
      Delta(delta),
      MaxIterations(maxit), 
      SinglePrecGrid4(_sp_grid4),
      SinglePrecGrid5(_sp_grid5)
    { 
    };

    void operator()(LinearOperatorBase<FieldD> &LinOpU, const FieldD &src, FieldD &psi) {

      std::cout << GridLogMessage << " Mixed precision reliable CG update wrapper operator() "<<std::endl;

      SchurOperatorD * SchurOpU = static_cast<SchurOperatorD *>(&LinOpU);
      assert(&(SchurOpU->_Mat)==&(LinOpD._Mat));

      precisionChange(FermOpF.Umu, FermOpD.Umu);

      pickCheckerboard(Even,FermOpF.UmuEven,FermOpF.Umu);
      pickCheckerboard(Odd ,FermOpF.UmuOdd ,FermOpF.Umu);

      ////////////////////////////////////////////////////////////////////////////////////
      // Make a mixed precision conjugate gradient
      ////////////////////////////////////////////////////////////////////////////////////

      ConjugateGradientReliableUpdate<FieldD,FieldF> MPCG(Tolerance,MaxIterations,Delta,SinglePrecGrid5,LinOpF,LinOpD);
      std::cout << GridLogMessage << "Calling mixed precision reliable update Conjugate Gradient" <<std::endl;
      MPCG(src,psi);
    }
  };



NAMESPACE_END(Grid);


//Do an offline comparison between the 2f action evaluated using the X-conj and G-parity operators, the latter with an X-conjugate vector
//It should be the same X-conjugate vector for both!
void actionCheck(typename XconjugateMobiusFermionD::ImplParams Params, double light_mass, double pv_mass, double M5, double mob_b, double mob_c, GridCartesian* UGridD, GridRedBlackCartesian* UrbGridD, GridCartesian* FGridD, GridRedBlackCartesian* FrbGridD,
		 GridParallelRNG &RNG5, LatticeGaugeField &Ud){
  typedef XconjugateMobiusFermionD XconjFermionActionD;
  typedef typename XconjFermionActionD::Impl_t XconjFermionImplPolicy;
  typedef typename XconjFermionActionD::FermionField FermionField1f;

  typedef GparityMobiusFermionD GparityFermionActionD;
  typedef typename GparityFermionActionD::Impl_t GparityFermionImplPolicy;
  typedef typename GparityFermionActionD::FermionField FermionField2f;
 
  XconjFermionActionD xconj_action_l(Ud,*FGridD,*FrbGridD,*UGridD,*UrbGridD,light_mass,M5,mob_b,mob_c,Params);
  XconjFermionActionD xconj_action_h(Ud,*FGridD,*FrbGridD,*UGridD,*UrbGridD,pv_mass,M5,mob_b,mob_c,Params);

  GparityFermionActionD::ImplParams gp_params;
  gp_params.twists = Params.twists;
  gp_params.dirichlet = Params.dirichlet;

  GparityFermionActionD gp_action_l(Ud,*FGridD,*FrbGridD,*UGridD,*UrbGridD,light_mass,M5,mob_b,mob_c,gp_params);
  GparityFermionActionD gp_action_h(Ud,*FGridD,*FrbGridD,*UGridD,*UrbGridD,pv_mass,M5,mob_b,mob_c,gp_params);
  
  typedef TwoFlavourEvenOddRatioPseudoFermionAction<XconjFermionImplPolicy> XconjTwoFlavorHMC;
  typedef TwoFlavourEvenOddRatioPseudoFermionAction<GparityFermionImplPolicy> GparityTwoFlavorHMC;

  ConjugateGradient<FermionField2f> CG2f(1e-10,10000);
  ConjugateGradient<FermionField1f> CG1f(1e-10,10000);
  
  XconjTwoFlavorHMC xconj_hmc(xconj_action_h, xconj_action_l, CG1f, CG1f, CG1f);
  GparityTwoFlavorHMC gp_hmc(gp_action_h, gp_action_l, CG2f, CG2f, CG2f);
  
  //Generate an X-conjugate 1f field
  FermionField1f rhoprime(FGridD);
  gaussian(RNG5,rhoprime);

  //rho = rho'/sqrt{2}
  FermionField2f rho(FGridD);
  static Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
  static Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
  static Gamma X = C*g5;
  
  FermionField1f tmp = -(X*conjugate(rhoprime));
  PokeIndex<GparityFlavourIndex>(rho, rhoprime, 0);
  PokeIndex<GparityFlavourIndex>(rho, tmp, 1);
  rho = rho * sqrt(0.5);

  xconj_hmc.setPhi(rhoprime);
  RealD S_xconj = xconj_hmc.S(Ud);
  
  gp_hmc.setPhi(rho);
  RealD S_gp = gp_hmc.S(Ud);

  std::cout << std::setprecision(16) << "actionCheck ml: " << light_mass << " mpv: " << pv_mass << " Xconj: " << S_xconj << " Gparity: " << S_gp << " diff: " << S_xconj - S_gp << std::endl;
}


  
  


int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  int threads = GridThread::GetThreads();
  // here make a routine to print all the relevant information on the run
  std::cout << GridLogMessage << "Grid is setup to use " << threads << " threads" << std::endl;

  std::string param_file = "params.xml";
  bool file_load_check = false;

  std::string serial_seeds = "1 2 3 4 5";
  std::string parallel_seeds = "6 7 8 9 10";

  int i=1;
  while(i < argc){
    std::string sarg(argv[i]);
    if(sarg == "--param_file"){
      assert(i!=argc-1);
      param_file = argv[i+1];
      i+=2;
    }else if(sarg == "--read_check"){ //check the fields load correctly and pass checksum/plaquette repro
      file_load_check = true;
      i++;
    }else if(sarg == "--set_seeds"){ //set the rng seeds. Expects two vector args, e.g.  --set_seeds 1.2.3.4 5.6.7.8
      assert(i < argc-2);
      std::vector<int> tmp;
      GridCmdOptionIntVector(argv[i+1],tmp);
      {
	std::stringstream ss;
	for(int j=0;j<tmp.size()-1;j++) ss << tmp[j] << " ";
	ss << tmp.back();
	serial_seeds = ss.str();
      }
      GridCmdOptionIntVector(argv[i+2],tmp);
      {
	std::stringstream ss;
	for(int j=0;j<tmp.size()-1;j++) ss << tmp[j] << " ";
	ss << tmp.back();
	parallel_seeds = ss.str();
      }
      i+=3;
      std::cout << GridLogMessage << "Set serial seeds to " << serial_seeds << std::endl;
      std::cout << GridLogMessage << "Set parallel seeds to " << parallel_seeds << std::endl;
      
    }else{
      i++;
    }
  }

  
  //Read the user parameters
  EvolParameters user_params;
  
  if(fileExists(param_file)){
    std::cout << GridLogMessage << " Reading " << param_file << std::endl;
    Grid::XmlReader rd(param_file);
    read(rd, "Params", user_params);
  }else if(!GlobalSharedMemory::WorldRank){
    std::cout << GridLogMessage << " File " << param_file << " does not exist" << std::endl;
    std::cout << GridLogMessage << " Writing xml template to " << param_file << ".templ" << std::endl;
    {
      Grid::XmlWriter wr(param_file + ".templ");
      write(wr, "Params", user_params);
    }
    std::cout << GridLogMessage << " Done" << std::endl;
    Grid_finalize();
    return 0;
  }

  //Check the parameters
  if(user_params.GparityDirs.size() != Nd-1){
    std::cerr << "Error in input parameters: expect GparityDirs to have size = " << Nd-1 << std::endl;
    exit(1);
  }
  for(int i=0;i<Nd-1;i++)
    if(user_params.GparityDirs[i] != 0 && user_params.GparityDirs[i] != 1){
      std::cerr << "Error in input parameters: expect GparityDirs values to be 0 (periodic) or 1 (G-parity)" << std::endl;
      exit(1);
    }


  typedef XconjugateMobiusEOFAFermionD EOFAactionD;
  typedef XconjugateMobiusFermionD FermionActionD;
  typedef typename FermionActionD::Impl_t FermionImplPolicyD;
  typedef typename FermionActionD::FermionField FermionFieldD;

  typedef XconjugateMobiusEOFAFermionF EOFAactionF;
  typedef XconjugateMobiusFermionF FermionActionF;
  typedef typename FermionActionF::Impl_t FermionImplPolicyF;
  typedef typename FermionActionF::FermionField FermionFieldF;

  typedef TwoFlavourEvenOddRatioPseudoFermionAction<FermionImplPolicyD> TwoFlavorHMCD;
  typedef TwoFlavourEvenOddRatioPseudoFermionAction<FermionImplPolicyF> TwoFlavorHMCF;

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  IntegratorParameters MD;
  typedef ConjugateHMCRunnerD<MinimumNorm2> HMCWrapper; //NB: This is the "Omelyan integrator"
  MD.name    = std::string("MinimumNorm2");

  // typedef ConjugateHMCRunnerD<ForceGradient> HMCWrapper;
  // MD.name    = std::string("ForceGradient");
  
  MD.MDsteps = user_params.Steps;
  MD.trajL   = user_params.TrajectoryLength;

  typedef HMCWrapper::ImplPolicy GaugeImplPolicy;
  
  HMCparameters HMCparams;
  HMCparams.StartTrajectory  = user_params.StartTrajectory;
  HMCparams.Trajectories     = user_params.Trajectories;
  HMCparams.NoMetropolisUntil= 0;
  HMCparams.StartingType     = user_params.StartingType;
  HMCparams.MetropolisTest = user_params.MetropolisTest;
  HMCparams.MD = MD;
  HMCWrapper TheHMC(HMCparams);

  // Grid from the command line arguments --grid and --mpi
  TheHMC.Resources.AddFourDimGrid("gauge"); // use default simd lanes decomposition

  CheckpointerParameters CPparams;
  CPparams.config_prefix = "ckpoint_lat";
  CPparams.rng_prefix    = "ckpoint_rng";
  CPparams.saveInterval  = user_params.SaveInterval;
  CPparams.format        = "IEEE64BIG";
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);

  //Note that checkpointing saves the RNG state so that this initialization is required only for the very first configuration
  RNGModuleParameters RNGpar;
  RNGpar.serial_seeds = serial_seeds;
  RNGpar.parallel_seeds = parallel_seeds;
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  typedef PlaquetteMod<GaugeImplPolicy> PlaqObs;
  TheHMC.Resources.AddObservable<PlaqObs>();
  
  const int Ls      = 8;
  Real beta         = 2.13;
  Real light_mass   = 0.01;
  Real strange_mass = 0.032;
  Real pv_mass      = 1.0;
  RealD M5  = 1.8;
  RealD mobius_scale = 2.; //b+c  -> same as Ls=16 DWF

  RealD mob_bmc = 1.0;
  RealD mob_b = (mobius_scale + mob_bmc)/2.;
  RealD mob_c = (mobius_scale - mob_bmc)/2.;

  std::cout << GridLogMessage
	    << "Ensemble parameters:" << std::endl
	    << "Ls=" << Ls << std::endl
	    << "beta=" << beta << std::endl
	    << "light_mass=" << light_mass << std::endl
	    << "strange_mass=" << strange_mass << std::endl
	    << "mobius_scale=" << mobius_scale << std::endl;
  
  //Setup the Grids
  auto UGridD   = TheHMC.Resources.GetCartesian();
  auto UrbGridD = TheHMC.Resources.GetRBCartesian();
  auto FGridD     = SpaceTimeGrid::makeFiveDimGrid(Ls,UGridD);
  auto FrbGridD   = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGridD);

  GridCartesian* UGridF = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplexF::Nsimd()), GridDefaultMpi());
  GridRedBlackCartesian* UrbGridF = SpaceTimeGrid::makeFourDimRedBlackGrid(UGridF);
  auto FGridF     = SpaceTimeGrid::makeFiveDimGrid(Ls,UGridF);
  auto FrbGridF   = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGridF);

  ConjugateIwasakiGaugeActionD GaugeAction(beta);

  // temporarily need a gauge field
  LatticeGaugeFieldD Ud(UGridD);
  LatticeGaugeFieldF Uf(UGridF);
 
  //Setup the BCs
  FermionActionD::ImplParams Params;
  for(int i=0;i<Nd-1;i++) Params.twists[i] = user_params.GparityDirs[i]; //G-parity directions
  Params.twists[Nd-1] = 1; //APBC in time direction
  Params.boundary_phase = 1.0; //X-conjugate
  
  std::vector<int> dirs4(Nd);
  for(int i=0;i<Nd-1;i++) dirs4[i] = user_params.GparityDirs[i];
  dirs4[Nd-1] = 0; //periodic gauge BC in time

  GaugeImplPolicy::setDirections(dirs4); //gauge BC

  //Run optional gauge field checksum checker and exit
  if(file_load_check){
    TheHMC.initializeGaugeFieldAndRNGs(Ud);
    std::cout << GridLogMessage << " Done" << std::endl;
    Grid_finalize();
    return 0;
  }


  typedef SchurDiagMooeeOperator<FermionActionD,FermionFieldD> SchurOpD;
  typedef SchurDiagMooeeOperator<FermionActionF,FermionFieldF> SchurOpF;  
  typedef MixedPrecisionReliableUpdateConjugateGradientOperatorFunction<FermionActionD,FermionActionF,SchurOpD,SchurOpF> MixedCG;

  ////////////////////////////////////
  // Collect actions
  ////////////////////////////////////
  ActionLevel<HMCWrapper::Field> Level1(1); //light quark + strange quark
  ActionLevel<HMCWrapper::Field> Level2(8); //gauge


  /////////////////////////////////////////////////////////////
  // Light 2f action
  /////////////////////////////////////////////////////////////

  std::vector<RealD> _2f_light_masses = { light_mass ,  0.03,   0.09,   0.27   };
  std::vector<RealD> _2f_pv_masses =    { 0.03       , 0.09,   0.27,   1.0      };
  int n_light_hsb = 4;
  assert(user_params.ratio_l.size() == n_light_hsb);
  
  for(int i=0;i<n_light_hsb;i++){
    RealD iml = _2f_light_masses[i];
    RealD ipv = _2f_pv_masses[i];

    FermionActionD* action_l_d = new FermionActionD(Ud,*FGridD,*FrbGridD,*UGridD,*UrbGridD,iml,M5,mob_b,mob_b-1.,Params);
    FermionActionD* action_h_d = new FermionActionD(Ud,*FGridD,*FrbGridD,*UGridD,*UrbGridD,ipv,M5,mob_b,mob_b-1.,Params);

    FermionActionF* action_l_f = new FermionActionF(Uf,*FGridF,*FrbGridF,*UGridF,*UrbGridF,iml,M5,mob_b,mob_b-1.,Params);
    FermionActionF* action_h_f = new FermionActionF(Uf,*FGridF,*FrbGridF,*UGridF,*UrbGridF,ipv,M5,mob_b,mob_b-1.,Params);

    SchurOpD* schurop_l_d = new SchurOpD(*action_l_d);
    SchurOpD* schurop_h_d = new SchurOpD(*action_h_d);

    SchurOpF* schurop_l_f = new SchurOpF(*action_l_f);
    SchurOpF* schurop_h_f = new SchurOpF(*action_h_f);

    MixedCG*  heatbath_solver = new MixedCG(user_params.ratio_l[i].action_tolerance, 0.1, 10000,UrbGridF,FrbGridF,*action_h_f,*action_h_d,*schurop_h_f,*schurop_h_d);  //inverts PV matrix
    MixedCG*  action_solver = new MixedCG(user_params.ratio_l[i].action_tolerance, 0.1, 10000,UrbGridF,FrbGridF,*action_l_f,*action_l_d,*schurop_l_f,*schurop_l_d);  //inverts light matrix
    MixedCG*  deriv_solver = new MixedCG(user_params.ratio_l[i].md_tolerance, 0.1, 10000,UrbGridF,FrbGridF,*action_l_f,*action_l_d,*schurop_l_f,*schurop_l_d);  //inverts light matrix

    TwoFlavorHMCD* pf_action = new TwoFlavorHMCD(*action_h_d, *action_l_d, *deriv_solver, *action_solver, *heatbath_solver);
  
    Level1.push_back(pf_action);
  }

  /////////////////////////////////////////////////////////////
  // Strange EOFA action
  // have to be careful with the parameters, cf. Test_dwf_gpforce_eofa.cc
  /////////////////////////////////////////////////////////////
  typedef SchurDiagMooeeOperator<EOFAactionD,FermionFieldD> EOFAschuropD;
  typedef SchurDiagMooeeOperator<EOFAactionF,FermionFieldF> EOFAschuropF;
  typedef ExactOneFlavourRatioMixedPrecHeatbathPseudoFermionAction<FermionImplPolicyD, FermionImplPolicyF> EOFAmixPrecPFaction;
  typedef MixedPrecisionConjugateGradientOperatorFunction<EOFAactionD, EOFAactionF, EOFAschuropD, EOFAschuropF> EOFA_mxCG;
  typedef MixedPrecisionReliableUpdateConjugateGradientOperatorFunction<EOFAactionD, EOFAactionF, EOFAschuropD, EOFAschuropF> EOFA_relupCG;

  EOFAmixPrecPFaction* EOFA;
  {
    RealD iml = strange_mass; 
    RealD ipv = 1.0;

    EOFAactionD* LopD = new EOFAactionD(Ud, *FGridD, *FrbGridD, *UGridD, *UrbGridD, iml, iml, ipv, 0.0, -1, M5, mob_b, mob_c, Params);
    EOFAactionF* LopF = new EOFAactionF(Uf, *FGridF, *FrbGridF, *UGridF, *UrbGridF, iml, iml, ipv, 0.0, -1, M5, mob_b, mob_c, Params);
    EOFAactionD* RopD = new EOFAactionD(Ud, *FGridD, *FrbGridD, *UGridD, *UrbGridD, ipv, iml, ipv, -1.0, 1, M5, mob_b, mob_c, Params);
    EOFAactionF* RopF = new EOFAactionF(Uf, *FGridF, *FrbGridF, *UGridF, *UrbGridF, ipv, iml, ipv, -1.0, 1, M5, mob_b, mob_c, Params);

    EOFAschuropD* linopL_D = new EOFAschuropD(*LopD);
    EOFAschuropD* linopR_D = new EOFAschuropD(*RopD);
    
    EOFAschuropF* linopL_F = new EOFAschuropF(*LopF);
    EOFAschuropF* linopR_F = new EOFAschuropF(*RopF);

    //Note reusing user_params.eofa_l.action(|md)_mixcg_inner_tolerance  as Delta for now
    EOFA_relupCG* ActionMCG_L = new EOFA_relupCG(user_params.eofa_s.action_tolerance, user_params.eofa_s.action_mixcg_inner_tolerance, 50000, UGridF, FrbGridF, *LopF, *LopD, *linopL_F, *linopL_D);
    EOFA_relupCG* ActionMCG_R = new EOFA_relupCG(user_params.eofa_s.action_tolerance, user_params.eofa_s.action_mixcg_inner_tolerance, 50000, UGridF, FrbGridF, *RopF, *RopD, *linopR_F, *linopR_D);

    EOFA_relupCG* DerivMCG_L = new EOFA_relupCG(user_params.eofa_s.md_tolerance, user_params.eofa_s.md_mixcg_inner_tolerance, 50000, UGridF, FrbGridF, *LopF, *LopD, *linopL_F, *linopL_D);
    EOFA_relupCG* DerivMCG_R = new EOFA_relupCG(user_params.eofa_s.md_tolerance, user_params.eofa_s.md_mixcg_inner_tolerance, 50000, UGridF, FrbGridF, *RopF, *RopD, *linopR_F, *linopR_D);

    EOFA = new EOFAmixPrecPFaction(*LopF, *RopF,
				   *LopD, *RopD, 
				   *ActionMCG_L, *ActionMCG_R, 
				   *ActionMCG_L, *ActionMCG_R, 
				   *DerivMCG_L, *DerivMCG_R, 
				   user_params.eofa_s.rat_params, true);
    Level1.push_back(EOFA);
  }

  
  /////////////////////////////////////////////////////////////
  // Gauge action
  /////////////////////////////////////////////////////////////
  Level2.push_back(&GaugeAction);

  TheHMC.TheAction.push_back(Level1);
  TheHMC.TheAction.push_back(Level2);
  std::cout << GridLogMessage << " Action complete "<< std::endl;

  //Action tuning
  bool 
    check_eofa=false, 
    upper_bound_eofa=false, lower_bound_eofa = false,
    action_check = false;

  for(int i=1;i<argc;i++){
    std::string sarg(argv[i]);
    if(sarg == "--check_eofa"){
      check_eofa = true;
    }
    else if(sarg == "--upper_bound_eofa"){
      upper_bound_eofa = true;
    }
    else if(sarg == "--lower_bound_eofa"){
      lower_bound_eofa = true;      
    }
    else if(sarg == "--action_check"){
      action_check = true;      
    }
  }
  if(check_eofa || upper_bound_eofa || lower_bound_eofa || action_check) {
    std::cout << GridLogMessage << "Running checks" << std::endl;
    TheHMC.initializeGaugeFieldAndRNGs(Ud);

    //std::cout << GridLogMessage << "EOFA action solver action tolerance outer=" << ActionMCG_L.Tolerance << " inner=" << ActionMCG_L.InnerTolerance << std::endl;
    //std::cout << GridLogMessage << "EOFA MD solver tolerance outer=" << DerivMCG_L.Tolerance << " inner=" << DerivMCG_L.InnerTolerance << std::endl;

    if(check_eofa) checkEOFA(*EOFA, FGridD, TheHMC.Resources.GetParallelRNG(), Ud);
    if(upper_bound_eofa) upperBoundEOFA(*EOFA, FGridD, TheHMC.Resources.GetParallelRNG(), Ud);
    if(lower_bound_eofa) lowerBoundEOFA(*EOFA, FGridD, TheHMC.Resources.GetParallelRNG(), Ud);
    if(action_check){
      std::cout << "Action check light mass" << std::endl;
      actionCheck(Params, light_mass, pv_mass, M5, mob_b, mob_c, UGridD, UrbGridD, FGridD, FrbGridD, TheHMC.Resources.GetParallelRNG(), Ud);
      std::cout << "Action check strange mass" << std::endl;
      actionCheck(Params, strange_mass, pv_mass, M5, mob_b, mob_c, UGridD, UrbGridD, FGridD, FrbGridD, TheHMC.Resources.GetParallelRNG(), Ud);
    }
    
    std::cout << GridLogMessage << " Done" << std::endl;
    Grid_finalize();
    return 0;
  }

  
  //Run the HMC
  std::cout << GridLogMessage << " Running the HMC "<< std::endl;
  TheHMC.Run();

  std::cout << GridLogMessage << " Done" << std::endl;
  Grid_finalize();
  return 0;
} // main
