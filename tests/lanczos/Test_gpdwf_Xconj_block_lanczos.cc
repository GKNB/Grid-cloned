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

  auto mpi_layout  = GridDefaultMpi();
  std::vector<int> mpi_split (Nd,1);
  int nsplit = 1;
  for(int i=0;i<Nd;i++) nsplit *= mpi_layout[i]/mpi_split[i];

  std::cout << GridLogMessage << "Number of subgrids: " << nsplit << std::endl;

  GridCartesian         * SUGrid = new GridCartesian(GridDefaultLatt(),
                                                    GridDefaultSimd(Nd,vComplex::Nsimd()),
                                                    mpi_split,
                                                    *UGrid);

  GridCartesian         * SFGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,SUGrid);
  GridRedBlackCartesian * SUrbGrid  = SpaceTimeGrid::makeFourDimRedBlackGrid(SUGrid);
  GridRedBlackCartesian * SFrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,SUGrid);

  LatticeGaugeField s_Umu(SUGrid);
  Grid_split  (Umu,s_Umu);

  //Setup the regular and X-conjugate actions
  RealD mass=0.01;
  RealD M5=1.8;
  RealD mob_b=1.5;
  std::vector<int> twists({1,1,1,0});  
    
  GparityMobiusFermionD ::ImplParams params_reg;
  params_reg.twists = twists;
  GparityMobiusFermionD action_reg(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_b-1.,params_reg);
  GparityMobiusFermionD s_action_reg(s_Umu,*SFGrid,*SFrbGrid,*SUGrid,*SUrbGrid,mass,M5,mob_b,mob_b-1.,params_reg);

  XconjugateMobiusFermionD ::ImplParams params_Xconj;
  params_Xconj.twists = twists;
  XconjugateMobiusFermionD action_Xconj(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_b-1.,params_Xconj);
  XconjugateMobiusFermionD s_action_Xconj(s_Umu,*SFGrid,*SFrbGrid,*SUGrid,*SUrbGrid,mass,M5,mob_b,mob_b-1.,params_Xconj);
  
  typedef GparityMobiusFermionD::FermionField TwoFlavorFermionField;
  typedef XconjugateMobiusFermionD::FermionField OneFlavorFermionField;

  SchurDiagMooeeOperator<GparityMobiusFermionD,TwoFlavorFermionField> HermOp_reg(action_reg);
  SchurDiagMooeeOperator<GparityMobiusFermionD,TwoFlavorFermionField> s_HermOp_reg(s_action_reg);

  SchurDiagMooeeOperator<XconjugateMobiusFermionD,OneFlavorFermionField> HermOp_Xconj(action_Xconj);
  SchurDiagMooeeOperator<XconjugateMobiusFermionD,OneFlavorFermionField> s_HermOp_Xconj(s_action_Xconj);

  //Defaults
  int Nstop = 32; 
  int Nk = 32; //Has to be divisible by Nu=4
  int Np = 8; //also has to be divisible by Nu
  int ord = 81;
  RealD lo = 10.0;
  RealD hi = 88.0;

  int Nu = nsplit; //set equal to number of splits
  
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

  int MaxIterBlock = 10; //this should be relatively small because Nm = Nk + Np*MaxIter
  int Nconv_test_skip = 4;
  int Nm_block = Nk + Np*MaxIterBlock;

  ImplicitlyRestartedBlockLanczos<TwoFlavorFermionField> IRL_block_reg(HermOp_reg, s_HermOp_reg, FrbGrid, SFrbGrid, nsplit, 
								       Cheby_reg, Nstop, Nconv_test_skip,
								       Nu, Nk, Nm_block, resid * 100, MaxIterBlock, IRBLdiagonaliseWithEigen);
  //note factor of 100 on resid is because the regular IRL normalizes the residual against the largest eval ( O(100) here ) whereas block Lanczos does not
  ImplicitlyRestartedBlockLanczos<OneFlavorFermionField> IRL_block_Xconj(HermOp_Xconj, s_HermOp_Xconj, FrbGrid, SFrbGrid, nsplit, 
									 Cheby_Xconj, Nstop, Nconv_test_skip,
									 Nu, Nk, Nm_block, resid * 100, MaxIterBlock, IRBLdiagonaliseWithEigen, inner_Xconj);

  std::vector<RealD> eval_reg(Nm);
  std::vector<RealD> eval_reg_block(Nm_block);
  std::vector<RealD> eval_Xconj(Nm);
  std::vector<RealD> eval_Xconj_block(Nm_block);

  std::vector<TwoFlavorFermionField> evec_reg(Nm,FrbGrid);
  std::vector<TwoFlavorFermionField> evec_reg_block(Nm_block,FrbGrid);
  std::vector<OneFlavorFermionField> evec_Xconj(Nm,FrbGrid);
  std::vector<OneFlavorFermionField> evec_Xconj_block(Nm_block,FrbGrid);

  TwoFlavorFermionField    src_reg(FrbGrid); 
  gaussian(RNG5rb,src_reg);

  OneFlavorFermionField    src_Xconj(FrbGrid); 
  gaussian(RNG5rb,src_Xconj);

  std::vector<TwoFlavorFermionField> src_reg_block(Nu,FrbGrid);
  for(int i=0;i<Nu;i++)
    gaussian(RNG5rb,src_reg_block[i]);

  std::vector<OneFlavorFermionField> src_Xconj_block(Nu,FrbGrid);
  for(int i=0;i<Nu;i++)
    gaussian(RNG5rb,src_Xconj_block[i]);
 
  int Nconv_Xconj;
  IRL_Xconj.calc(eval_Xconj,evec_Xconj,src_Xconj,Nconv_Xconj);

  int Nconv_Xconj_block;
  IRL_block_Xconj.calc(eval_Xconj_block,evec_Xconj_block,src_Xconj_block,Nconv_Xconj_block,LanczosType::rbl);

  //Block has different ordering
  std::reverse(eval_Xconj_block.begin(),eval_Xconj_block.end());
  std::reverse(evec_Xconj_block.begin(),evec_Xconj_block.end());

  int Nconv_reg;
  IRL_reg.calc(eval_reg,evec_reg,src_reg,Nconv_reg);

  int Nconv_reg_block;
  IRL_block_reg.calc(eval_reg_block,evec_reg_block,src_reg_block,Nconv_reg_block,LanczosType::rbl);

  //Block has different ordering
  std::reverse(eval_reg_block.begin(),eval_reg_block.end());
  std::reverse(evec_reg_block.begin(),evec_reg_block.end());
  
  std::cout << "Comparing evals block vs regular for two-flavor operator: " << std::endl;
  //for(int i=0;i<std::min(Nconv_reg, Nconv_reg_block);i++){
  for(int i=0;i<Nstop;i++){
    std::cout << eval_reg_block[i] << " " << eval_reg[i] << " diff: " << eval_reg_block[i] - eval_reg[i] << std::endl;
  }

  std::cout << "Comparing evals Xconj vs regular for unblocked Lanczos: " << std::endl;
  for(int i=0;i<std::min(Nconv_reg, Nconv_Xconj);i++){
    std::cout << eval_Xconj[i] << " " << eval_reg[i] << " diff: " << eval_Xconj[i] - eval_reg[i] << std::endl;
  }

  std::cout << "Comparing evals block vs regular for one-flavor operator: " << std::endl;
  for(int i=0;i<Nstop;i++){
    std::cout << eval_Xconj_block[i] << " " << eval_Xconj[i] << " diff: " << eval_Xconj_block[i] - eval_Xconj[i] << std::endl;
  }

  std::cout << "Comparing evecs block vs regular for one-flavor operator: " << std::endl;
  for(int i=0;i<Nstop;i++){
    OneFlavorFermionField tmp(FrbGrid);
    tmp = evec_Xconj_block[i] - evec_Xconj[i];
    std::cout << i << " " << norm2(tmp) << std::endl;
  }

  std::cout << "Done" << std::endl;
  Grid_finalize();
}
