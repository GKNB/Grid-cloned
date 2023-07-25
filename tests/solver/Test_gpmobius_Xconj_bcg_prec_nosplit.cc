   /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/Test_dwf_mrhs_cg.cc

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

#include <Grid/Grid.h>

#include <Grid/algorithms/iterative/BlockConjugateGradient.h>
using namespace std;
using namespace Grid;
 
typedef typename GparityMobiusFermionD::FermionField FermionField2f;
typedef typename XconjugateMobiusFermionD::FermionField FermionField1f;

template<typename Action>
typename Action::FermionField solveCG(const typename Action::FermionField &src_4d, Action &action){
  typedef typename Action::FermionField FermionField;
  ConjugateGradient<FermionField> cg(1e-08,10000);
  SchurRedBlackDiagMooeeSolve<FermionField> solver(cg);
  
  GridBase *UGrid = action.GaugeGrid();
  GridBase *FGrid = action.FermionGrid();
  GridBase *FrbGrid = action.FermionRedBlackGrid();
  
  FermionField src_5d(FGrid), src_5d_e(FrbGrid), src_5d_o(FrbGrid), sol_5d_o(FrbGrid), sol_5d(FGrid), sol_4d(UGrid);

  sol_5d_o = Zero();

  action.ImportPhysicalFermionSource(src_4d, src_5d);
  solver.RedBlackSource(action, src_5d, src_5d_e, src_5d_o);
  solver.RedBlackSolve(action, src_5d_o, sol_5d_o);
  solver.RedBlackSolution(action, sol_5d_o, src_5d_e, sol_5d);
  action.ExportPhysicalFermionSolution(sol_5d, sol_4d);
  return sol_4d;
}

template<typename Action>
std::vector<typename Action::FermionField> solveBlockCG(const std::vector<typename Action::FermionField> &src_4d, Action &action){
  typedef typename Action::FermionField FermionField;
  BlockConjugateGradient<FermionField> cg(BlockCGrQVec,0,1e-08,100000);
  SchurRedBlackDiagMooeeSolve<FermionField> solver(cg);
  
  GridBase *UGrid = action.GaugeGrid();
  GridBase *FGrid = action.FermionGrid();
  GridBase *FrbGrid = action.FermionRedBlackGrid();
  
  int N = src_4d.size();

  std::vector<FermionField> src_5d(N,FGrid), src_5d_e(N,FrbGrid), src_5d_o(N,FrbGrid), sol_5d_o(N,FrbGrid), sol_5d(N,FGrid), sol_4d(N,UGrid);

  for(int i=0;i<N;i++){
    sol_5d_o[i] = Zero();
    action.ImportPhysicalFermionSource(src_4d[i], src_5d[i]);
    solver.RedBlackSource(action, src_5d[i], src_5d_e[i], src_5d_o[i]);
  }
  solver.RedBlackSolve(action, src_5d_o, sol_5d_o);
  for(int i=0;i<N;i++){
    solver.RedBlackSolution(action, sol_5d_o[i], src_5d_e[i], sol_5d[i]);
    action.ExportPhysicalFermionSolution(sol_5d[i], sol_4d[i]);
  }
  return sol_4d;
}

FermionField2f boostXconj(const FermionField1f &f){
  static Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
  static Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
  static Gamma X = C*g5;

  FermionField2f out(f.Grid());
  out.Checkerboard() = f.Checkerboard();

  PokeIndex<GparityFlavourIndex>(out, f, 0);
  FermionField1f tmp(f.Grid());
  tmp = -(X*conjugate(f));
  PokeIndex<GparityFlavourIndex>(out, tmp, 1);  
  return out;
}

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);
 
  int Ls=16;
  bool load_config = false;
  std::string load_config_file;
  RealD mass=0.01;  
  RealD bpc=2.0;
  bool is_cps_cfg = false;

  for(int i=1;i<argc;i++){
    std::string arg = argv[i];
    if(arg == "--Ls"){
      Ls = std::stoi(argv[i+1]);
      std::cout << GridLogMessage << "Set Ls to " << Ls << std::endl;
    }else if(arg == "--mass"){
      std::stringstream ss; ss << argv[i+1]; ss >> mass;
      std::cout << GridLogMessage << "Set mass to " << mass << std::endl;
    }else if(arg == "--bplusc"){
      std::stringstream ss; ss << argv[i+1]; ss >> bpc;
      std::cout << GridLogMessage << "Set Mobius b+c to " << bpc << std::endl;
    }else if(arg == "--load_config"){
      load_config = true;
      load_config_file = argv[i+1];
      std::cout << GridLogMessage << "Using configuration " << load_config_file << std::endl;
    }else if(arg == "--is_cps_config"){
      is_cps_cfg = true;
      std::cout << GridLogMessage << "Configuration is a CPS config" << std::endl;
    }
  }  

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

  if(load_config) { 
    FieldMetaData header;
    std::string file(load_config_file);
    if(is_cps_cfg) NerscIO::exitOnReadPlaquetteMismatch() = false;
    typedef GaugeStatistics<ConjugateGimplD> GaugeStats;
    NerscIO::readConfiguration<GaugeStats>(Umu,header,file);
    if(is_cps_cfg) NerscIO::exitOnReadPlaquetteMismatch() = true;
    std::cout << GridLogMessage << " Config "<<file<<" successfully read" <<std::endl;
  } else{
    SU<Nc>::HotConfiguration(RNG4, Umu);
    std::cout << GridLogMessage << "Intialised the HOT Gauge Field"<<std::endl;
  }

  Gamma C = Gamma(Gamma::Algebra::MinusGammaY) * Gamma(Gamma::Algebra::GammaT);
  Gamma g5 = Gamma(Gamma::Algebra::Gamma5);
  Gamma X = C*g5;
  
  //Set up a regular MDWF action instance as well as X-conj and Xbar-conj versions
  RealD M5=1.8;
  RealD bmc = 1.0;
  RealD mob_b = 0.5*(bpc + bmc);

  GparityMobiusFermionD ::ImplParams params;
  std::vector<int> twists({1,1,1,0});
  params.twists = twists;
    
  GparityMobiusFermionD reg_action(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_b-bmc,params);

  XconjugateMobiusFermionD::ImplParams xparams;
  xparams.twists = twists;
  xparams.boundary_phase = 1.0;
  
  XconjugateMobiusFermionD xconj_action(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,mob_b,mob_b-bmc,xparams);

  FermionField2f src_2f(UGrid);
  std::cout.precision(12);

  gaussian(RNG4, src_2f);
  
  //Break the source into two X-conjugate sources
  FermionField1f src_2f_f0 = PeekIndex<GparityFlavourIndex>(src_2f,0);
  FermionField1f src_2f_f1 = PeekIndex<GparityFlavourIndex>(src_2f,1);
  
  FermionField1f rho_1f = ComplexD(0.5)*( src_2f_f0 + X*conjugate(src_2f_f1) );
  FermionField1f tau_1f = ComplexD(0,-0.5)*( src_2f_f0 - X*conjugate(src_2f_f1) );

  //Solve with 2f matrix
  FermionField2f sol_2f_reg = solveCG(src_2f, reg_action);
  
  //Solve with 1f matrix, two solves
  FermionField1f sol_1f_rho = solveCG(rho_1f, xconj_action);
  FermionField1f sol_1f_tau = solveCG(tau_1f, xconj_action);
  
  FermionField2f sol_2f_Xconj = boostXconj(sol_1f_rho) + ComplexD(0,1)*boostXconj(sol_1f_tau);
  
  FermionField2f diff = sol_2f_Xconj - sol_2f_reg;
  std::cout << "Regular solve vs 2 independent X-conj solves: " << norm2(diff) << " (expect 0)" << std::endl;

  std::vector<FermionField2f> block_src_2f(2, UGrid);
  block_src_2f[0] = boostXconj(rho_1f);
  block_src_2f[1] = boostXconj(tau_1f);
  
  std::vector<FermionField2f> block_sol_2f = solveBlockCG(block_src_2f, reg_action);
  sol_2f_Xconj = block_sol_2f[0] + ComplexD(0,1)*block_sol_2f[1];
  
  diff = sol_2f_Xconj - sol_2f_reg;
  std::cout << "Regular solve vs block regular solve: " << norm2(diff) << " (expect 0)" << std::endl;

#if 0 //Doesn't yet work (inner product definition)
  std::vector<FermionField1f> block_src_1f(2, UGrid);
  block_src_1f[0] = rho_1f;
  block_src_1f[1] = tau_1f;
  
  std::vector<FermionField1f> block_sol_1f = solveBlockCG(block_src_1f, xconj_action);
  sol_2f_Xconj = boostXconj(block_sol_1f[0]) + ComplexD(0,1)*boostXconj(block_sol_1f[1]);
  
  diff = sol_2f_Xconj - sol_2f_reg;
  std::cout << "Regular solve vs block X-conj solve: " << norm2(diff) << " (expect 0)" << std::endl;
#endif

  Grid_finalize();
}
