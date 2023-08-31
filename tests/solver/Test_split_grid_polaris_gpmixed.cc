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

using namespace std;
using namespace Grid;
 ;

int main (int argc, char ** argv)
{
  typedef GparityMobiusFermionD DiracD;
  typedef GparityMobiusFermionF DiracF;
    
  typedef typename DiracD::FermionField FermionFieldD; 
  typedef typename DiracD::ComplexField ComplexFieldD; 
  typedef typename DiracF::FermionField FermionFieldF; 
  typedef typename DiracF::ComplexField ComplexFieldF; 
  
  Grid_init(&argc,&argv);

  const int Ls=12;
  double tolerance = 1e-20;
  Coordinate mpi_layout  = GridDefaultMpi();
  Coordinate mpi_split (mpi_layout.size(),1);
  
  for(int i=0;i<argc;i++){
    std::string sarg = argv[i];
    if(sarg == "--split"){
      GridCmdOptionIntVector(argv[i+1], mpi_split);
    }else if(sarg== "--tol"){
      tolerance = std::stod(argv[i+1]);
    }    
  }

  int nrhs = 1;
  for(int i=0;i<mpi_layout.size();i++) nrhs *= (mpi_layout[i]/mpi_split[i]);
 
  typename DiracD::ImplParams params;
  params.twists[0] = params.twists[1] = params.twists[2] = 1;
  params.twists[3] = 1; //APBC in t
  
  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layoutD = GridDefaultSimd(Nd,vComplexD::Nsimd());
  Coordinate simd_layoutF = GridDefaultSimd(Nd,vComplexF::Nsimd()); 

  GridCartesian         * UGridF   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), simd_layoutF ,GridDefaultMpi());
  GridCartesian         * FGridF   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGridF);
  GridRedBlackCartesian * UrbGridF  = SpaceTimeGrid::makeFourDimRedBlackGrid(UGridF);
  GridRedBlackCartesian * FrbGridF = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGridF);

  GridCartesian         * SUGridF = new GridCartesian(GridDefaultLatt(),
						      simd_layoutF,
						      mpi_split,
						      *UGridF); 
  GridCartesian         * SFGridF   = SpaceTimeGrid::makeFiveDimGrid(Ls,SUGridF);
  GridRedBlackCartesian * SUrbGridF  = SpaceTimeGrid::makeFourDimRedBlackGrid(SUGridF);
  GridRedBlackCartesian * SFrbGridF = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,SUGridF);


  GridCartesian         * UGridD   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), simd_layoutD ,GridDefaultMpi());
  GridCartesian         * FGridD   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGridD);
  GridRedBlackCartesian * UrbGridD  = SpaceTimeGrid::makeFourDimRedBlackGrid(UGridD);
  GridRedBlackCartesian * FrbGridD = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGridD);

  GridCartesian         * SUGridD = new GridCartesian(GridDefaultLatt(),
						      simd_layoutD,
						      mpi_split,
						      *UGridD); 
  GridCartesian         * SFGridD   = SpaceTimeGrid::makeFiveDimGrid(Ls,SUGridD);
  GridRedBlackCartesian * SUrbGridD  = SpaceTimeGrid::makeFourDimRedBlackGrid(SUGridD);
  GridRedBlackCartesian * SFrbGridD = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,SUGridD);

  
 


  ///////////////////////////////////////////////
  // Set up the problem as a 4d spreadout job
  ///////////////////////////////////////////////
  std::vector<int> seeds({1,2,3,4});

  GridParallelRNG pRNG(UGridD);  pRNG.SeedFixedIntegers(seeds);
  GridParallelRNG pRNG5(FGridD);  pRNG5.SeedFixedIntegers(seeds);
  FermionFieldD    src_full(FGridD);
  std::vector<FermionFieldD> src(nrhs,FrbGridD);
  for(int s=0;s<nrhs;s++){
    random(pRNG5,src_full);
    pickCheckerboard(Odd, src[s], src_full);
  }  
  std::vector<FermionFieldD> result(nrhs,FrbGridD);

  for(int s=0;s<nrhs;s++) result[s]=Zero();

  LatticeGaugeFieldD UmuD(UGridD); SU<Nc>::HotConfiguration(pRNG,UmuD);
  
  /////////////////
  // MPI only sends
  /////////////////
  LatticeGaugeFieldD s_UmuD(SUGridD);
  FermionFieldD s_src(SFrbGridD);
  FermionFieldD s_tmp(SFrbGridD);
  FermionFieldD s_res(SFrbGridD);

  ///////////////////////////////////////////////////////////////
  // split the source out using MPI instead of I/O
  ///////////////////////////////////////////////////////////////
  Grid_split  (UmuD,s_UmuD);
  Grid_split  (src,s_src);

  LatticeGaugeFieldF s_UmuF(SUGridF);
  precisionChange(s_UmuF,s_UmuD);
  
  ///////////////////////////////////////////////////////////////
  // Set up N-solvers as trivially parallel
  ///////////////////////////////////////////////////////////////
  RealD mass=0.01;
  RealD M5=1.8;

  RealD bpc = 2.0, bmc= 1.0;
  RealD b = (bpc + bmc)/2., c = (bpc - bmc)/2.;

  DiracD DdwfD(s_UmuD,*SFGridD,*SFrbGridD,*SUGridD,*SUrbGridD,mass,M5,b,c,params);
  DiracF DdwfF(s_UmuF,*SFGridF,*SFrbGridF,*SUGridF,*SUrbGridF,mass,M5,b,c,params);

  SchurDiagMooeeOperator<DiracD,FermionFieldD> HermOpD(DdwfD);
  SchurDiagMooeeOperator<DiracF,FermionFieldF> HermOpF(DdwfF);
  
  std::cout << GridLogMessage << "****************************************************************** "<<std::endl;
  std::cout << GridLogMessage << " Calling DWF CG "<<std::endl;
  std::cout << GridLogMessage << "****************************************************************** "<<std::endl;

  ConjugateGradientReliableUpdate<FermionFieldD,FermionFieldF> CG(tolerance, 10000000, 0.1, SFrbGridF, HermOpF, HermOpD);

  s_res = Zero();
  CG(s_src,s_res);

  /////////////////////////////////////////////////////////////
  // Gather and residual check on the results
  /////////////////////////////////////////////////////////////
  std::cout << GridLogMessage<< "Unsplitting the result"<<std::endl;
  Grid_unsplit(result,s_res);
  
  Grid_finalize();
}
