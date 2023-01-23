#include <Grid/Grid.h>

using namespace std;
using namespace Grid;
 ;

struct LanczosParams2 : Serializable {
public:
  GRID_SERIALIZABLE_CLASS_MEMBERS(LanczosParams2,
				  RealD, lo,
				  RealD, hi,
				  int, ord,
				  int, Nk,
				  int, Np,
				  int, Nstop);
};


int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  bool read_params = false;
  LanczosParams2 params;
  for(int i=1;i<argc;i++){
    std::string sarg = argv[i];
    if(sarg == "-params"){
      std::cout << "Reading params from " << argv[i+1] << std::endl;
      {
	XmlReader reader(argv[i+1]);
	read(reader, "Params", params);
      }
      read_params = true;
      std::cout << "Got params:" << std::endl << params << std::endl;
    }
  }

  const int Ls=8;

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);
  printf("UGrid=%p UrbGrid=%p FGrid=%p FrbGrid=%p\n",UGrid,UrbGrid,FGrid,FrbGrid);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  GridParallelRNG          RNG5(FGrid);  RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG          RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);
  GridParallelRNG          RNG5rb(FrbGrid);  RNG5.SeedFixedIntegers(seeds5);

  LatticeGaugeField Umu(UGrid); 
  SU<Nc>::HotConfiguration(RNG4, Umu);

  //Setup the regular and X-conjugate actions
  RealD mass=0.01;
  RealD M5=1.8;
  RealD mob_b=1.5;
  std::vector<int> twists({1,1,1,0});  
    
  GparityMobiusFermionD ::ImplParams params_reg;
  params_reg.twists = twists;
  GparityMobiusFermionR action_reg(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_b-1.,params_reg);

  XconjugateMobiusFermionR ::ImplParams params_Xconj;
  params_Xconj.twists = twists;
  XconjugateMobiusFermionR action_Xconj(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_b-1.,params_Xconj);
  
  typedef GparityMobiusFermionD::FermionField TwoFlavorFermionField;
  typedef XconjugateMobiusFermionR::FermionField OneFlavorFermionField;

  SchurDiagMooeeOperator<GparityMobiusFermionR,TwoFlavorFermionField> HermOp_reg(action_reg);
  SchurDiagMooeeOperator<XconjugateMobiusFermionR,OneFlavorFermionField> HermOp_Xconj(action_Xconj);

  //Defaults
  int Nstop = 30;
  int Nk = 30;
  int Np = 5;
  int ord = 81;
  RealD lo = 10.0;
  RealD hi = 88.0;
  
  //User override
  if(read_params){
    lo = params.lo;
    hi = params.hi;
    Nstop = params.Nstop;
    Nk = params.Nk;
    Np = params.Np;
    ord = params.ord;
  }

  const int Nm = Nk+Np;
  const int MaxIt= 10000;
  RealD resid = 1.0e-8;

  //Setup the IRL
  Chebyshev<TwoFlavorFermionField> Cheby_reg(lo,hi,ord);
  FunctionHermOp<TwoFlavorFermionField> OpCheby_reg(Cheby_reg,HermOp_reg);
  PlainHermOp<TwoFlavorFermionField> Op_reg(HermOp_reg);
  ImplicitlyRestartedLanczos<TwoFlavorFermionField> IRL_reg(OpCheby_reg,Op_reg,Nstop,Nk,Nm,resid,MaxIt);


  Chebyshev<OneFlavorFermionField> Cheby_Xconj(lo,hi,ord);
  FunctionHermOp<OneFlavorFermionField> OpCheby_Xconj(Cheby_Xconj,HermOp_Xconj);
  PlainHermOp<OneFlavorFermionField> Op_Xconj(HermOp_Xconj);

  innerProductImplementationXconjugate<OneFlavorFermionField> inner_Xconj;
  ImplicitlyRestartedLanczosHermOpTester<OneFlavorFermionField> tester_Xconj(Op_Xconj, inner_Xconj);
  ImplicitlyRestartedLanczos<OneFlavorFermionField> IRL_Xconj(OpCheby_Xconj,Op_Xconj,tester_Xconj,inner_Xconj,Nstop,Nk,Nm,resid,MaxIt);
 
  std::vector<RealD> eval_reg(Nm);
  std::vector<RealD> eval_Xconj(Nm);

  std::vector<TwoFlavorFermionField> evec_reg(Nm,FrbGrid);
  std::vector<OneFlavorFermionField> evec_Xconj(Nm,FrbGrid);

  TwoFlavorFermionField    src_reg(FrbGrid); 
  gaussian(RNG5rb,src_reg);

  OneFlavorFermionField    src_Xconj(FrbGrid); 
  gaussian(RNG5rb,src_Xconj);

  int Nconv_Xconj;
  IRL_Xconj.calc(eval_Xconj,evec_Xconj,src_Xconj,Nconv_Xconj);

  int Nconv_reg;
  IRL_reg.calc(eval_reg,evec_reg,src_reg,Nconv_reg);

  std::cout << "Converged regular: " << Nconv_reg << " Xconj: " << Nconv_Xconj << std::endl;
  std::cout << "Comparing evals: " << std::endl;
  for(int i=0;i<std::min(Nconv_reg, Nconv_Xconj);i++){
    std::cout << eval_Xconj[i] << " " << eval_reg[i] << " diff: " << eval_Xconj[i] - eval_reg[i] << std::endl;
  }

  Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
  Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
  Gamma X = C*g5;

  std::cout << "Comparing eigenvectors: " << std::endl;
  for(int i=0;i<std::min(Nconv_reg, Nconv_Xconj);i++){
    OneFlavorFermionField tmp(FrbGrid);
    HermOp_Xconj.HermOp(evec_Xconj[i], tmp);
    tmp = tmp - eval_Xconj[i]*evec_Xconj[i];
    std::cout << "Test Xconj evec is an evec (expect 0): " << norm2(tmp) << "  and norm2 of evec should be 1/2: " << norm2(evec_Xconj[i]) << std::endl;

    TwoFlavorFermionField tmp2f(FrbGrid);
    HermOp_reg.HermOp(evec_reg[i], tmp2f);
    tmp2f = tmp2f - eval_reg[i]*evec_reg[i];
    std::cout << "Test Gparity evec is an evec (expect 0): " << norm2(tmp2f) << "  and norm2 of evec should be 1: " << norm2(evec_reg[i]) << std::endl;

    //We need to phase-rotate the regular evecs into X-conjugate vectors
    OneFlavorFermionField v0 = PeekIndex<GparityFlavourIndex>(evec_reg[i],0);
    OneFlavorFermionField v1 = PeekIndex<GparityFlavourIndex>(evec_reg[i],1);
    OneFlavorFermionField Xv1star = X*conjugate(v1);
    ComplexD z = innerProduct(v0, Xv1star);
    ComplexD alpha = ComplexD(0.5)/z;

    TwoFlavorFermionField w = 1./sqrt(alpha) * evec_reg[i];
    //w should be X-conjugate; check
    OneFlavorFermionField w0 = PeekIndex<GparityFlavourIndex>(w,0);
    OneFlavorFermionField w1 = PeekIndex<GparityFlavourIndex>(w,1);
    tmp = w1 + X*conjugate(w0);
    std::cout << "Converted regular evec to X-conjugate: check (expect 0): " << norm2(tmp) << std::endl;

    tmp = w0 - evec_Xconj[i];
    OneFlavorFermionField tmp2 = w0 + evec_Xconj[i];
    std::cout << "Evec " << i << " difference: " << norm2(tmp) << " sum: " << norm2(tmp2) << std::endl;
  }



  std::cout << "Done" << std::endl;
  Grid_finalize();
}
