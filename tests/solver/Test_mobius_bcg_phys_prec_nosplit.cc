   /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./tests/Test_dwf_mrhs_cg.cc

    Copyright (C) 2015

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
 
int main (int argc, char ** argv)
{
  typedef typename DomainWallFermionD::FermionField FermionField; 
  typedef typename DomainWallFermionD::ComplexField ComplexField; 
  typename DomainWallFermionD::ImplParams params; 

  Grid_init(&argc,&argv);

  int Ls=16;
  int nrhs = 4;
  bool load_config = false;
  std::string load_config_file;
  RealD mass=0.01;
  RealD bpc=1.0; //b+c
    

  for(int i=1;i<argc;i++){
    std::string arg = argv[i];
    if(arg == "--Ls"){
      Ls = std::stoi(argv[i+1]);
      std::cout << GridLogMessage << "Set Ls to " << Ls << std::endl;
    }else if(arg == "--nrhs"){
      nrhs = std::stoi(argv[i+1]);
      std::cout << GridLogMessage << "Set nrhs to " << nrhs << std::endl;
    }else if(arg == "--mass"){
      std::stringstream ss; ss << argv[i+1]; ss >> mass;
      std::cout << GridLogMessage << "Set mass to " << mass << std::endl;
    }else if(arg == "--load_config"){
      load_config = true;
      load_config_file = argv[i+1];
      std::cout << GridLogMessage << "Using configuration " << load_config_file << std::endl;
    }else if(arg == "--bplusc"){
      std::stringstream ss; ss << argv[i+1]; ss >> bpc;
      std::cout << GridLogMessage << "Set Mobius b+c to " << bpc << std::endl;
    }
  }  

  auto latt_size   = GridDefaultLatt();
  auto simd_layout = GridDefaultSimd(Nd,vComplex::Nsimd());
  auto mpi_layout  = GridDefaultMpi();

  std::vector<Complex> boundary_phases(Nd,1.);
  boundary_phases[Nd-1]=-1.;
  params.boundary_phases = boundary_phases;

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), 
								   GridDefaultSimd(Nd,vComplex::Nsimd()),
								   GridDefaultMpi());
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * rbGrid  = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);
 
  double stp = 1.e-8;

  ///////////////////////////////////////////////
  // Set up the problem as a 4d spreadout job
  ///////////////////////////////////////////////
  std::vector<int> seeds({1,2,3,4});

  std::vector<FermionField> src4d(nrhs,UGrid);
  std::vector<FermionField> src(nrhs,FGrid);
  std::vector<FermionField> result(nrhs,FGrid);
  FermionField tmp(FGrid);
  std::cout << GridLogMessage << "Made the Fermion Fields"<<std::endl;

  for(int s=0;s<nrhs;s++) result[s]=Zero();
  GridParallelRNG pRNG5(FGrid);  pRNG5.SeedFixedIntegers(seeds);
  GridParallelRNG pRNG4(UGrid);  pRNG4.SeedFixedIntegers(seeds);
  for(int s=0;s<nrhs;s++) {
    random(pRNG4,src4d[s]);    
    std::cout << GridLogMessage << " src ["<<s<<"] "<<norm2(src[s])<<std::endl;
  }

  std::cout << GridLogMessage << "Intialised the Fermion Fields"<<std::endl;

  LatticeGaugeField Umu(UGrid); 

  if(load_config) { 
    FieldMetaData header;
    std::string file(load_config_file);
    NerscIO::readConfiguration(Umu,header,file);
    std::cout << GridLogMessage << " Config "<<file<<" successfully read" <<std::endl;
  } else{
    GridParallelRNG pRNG(UGrid );  
    pRNG.SeedFixedIntegers(seeds);
    SU<Nc>::HotConfiguration(pRNG,Umu);
    std::cout << GridLogMessage << "Intialised the HOT Gauge Field"<<std::endl;
  }

  ///////////////////////////////////////////////////////////////
  // Set up N-solvers as trivially parallel
  ///////////////////////////////////////////////////////////////
  std::cout << GridLogMessage << " Building the solvers"<<std::endl;
  RealD M5=1.8;
  //DomainWallFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*rbGrid,mass,M5,params);

  RealD bmc = 1.0;
  RealD mob_b = 0.5*(bpc + bmc);

  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*rbGrid,mass,M5,mob_b,mob_b-bmc,params);
  for(int i=0;i<nrhs;i++)
    Ddwf.ImportPhysicalFermionSource(src4d[i],src[i]);
  
  std::cout << GridLogMessage << "****************************************************************** "<<std::endl;
  std::cout << GridLogMessage << " Calling DWF CG "<<std::endl;
  std::cout << GridLogMessage << "****************************************************************** "<<std::endl;

  //MdagMLinearOperator<DomainWallFermionD,FermionField> HermOp(Ddwf);
  //MdagMLinearOperator<MobiusFermionD,FermionField> HermOp(Ddwf);
  ConjugateGradient<FermionField> CG((stp),100000);
  SchurRedBlackDiagTwoSolve<FermionField> SchurSolverSingle(CG);

  for(int rhs=0;rhs<1;rhs++){
    result[rhs] = Zero();
    //CG(HermOp,src[rhs],result[rhs]);
    SchurSolverSingle(Ddwf,src[rhs],result[rhs]);    
  }

  for(int rhs=0;rhs<1;rhs++){
    std::cout << " Result["<<rhs<<"] norm = "<<norm2(result[rhs])<<std::endl;
  }

  /////////////////////////////////////////////////////////////
  // Try block CG
  /////////////////////////////////////////////////////////////
  int blockDim = 0;//not used for BlockCGVec
  for(int s=0;s<nrhs;s++){
    result[s]=Zero();
  }


  {
    BlockConjugateGradient<FermionField>    BCGV  (BlockCGrQVec,blockDim,stp,100000);
    SchurRedBlackDiagTwoSolve<FermionField> SchurSolver(BCGV);
    SchurSolver(Ddwf,src,result);
  }
  
  for(int rhs=0;rhs<nrhs;rhs++){
    std::cout << " Result["<<rhs<<"] norm = "<<norm2(result[rhs])<<std::endl;
  }

  Grid_finalize();
}
