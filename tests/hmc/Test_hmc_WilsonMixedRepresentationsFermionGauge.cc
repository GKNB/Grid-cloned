/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/Test_hmc_WilsonAdjointFermionGauge.cc

Copyright (C) 2015

Author: Peter Boyle <pabobyle@ph.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: neo <cossu@post.kek.jp>
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
#include "Grid/Grid.h"



#ifdef ENABLE_FERMION_REPS

int main(int argc, char **argv) {
#ifndef GRID_CUDA
  using namespace Grid;


  // Here change the allowed (higher) representations
  typedef Representations< FundamentalRepresentation, AdjointRepresentation , TwoIndexSymmetricRepresentation> TheRepresentations;

  Grid_init(&argc, &argv);
  int threads = GridThread::GetThreads();
  // here make a routine to print all the relevant information on the run
  std::cout << GridLogMessage << "Grid is setup to use " << threads << " threads" << std::endl;

   // Typedefs to simplify notation
  typedef GenericHMCRunnerHirep<TheRepresentations, MinimumNorm2> HMCWrapper;

  typedef WilsonAdjImplR AdjImplPolicy; // gauge field implemetation for the pseudofermions
  typedef WilsonAdjFermionD AdjFermionAction; // type of lattice fermions (Wilson, DW, ...)
  typedef WilsonTwoIndexSymmetricImplR SymmImplPolicy; 
  typedef WilsonTwoIndexSymmetricFermionD SymmFermionAction; 


  typedef typename AdjFermionAction::FermionField AdjFermionField;
  typedef typename SymmFermionAction::FermionField SymmFermionField;

  //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  HMCWrapper TheHMC;

  // Grid from the command line
  TheHMC.Resources.AddFourDimGrid("gauge");
  // Possibile to create the module by hand 
  // hardcoding parameters or using a Reader


  // Checkpointer definition
  CheckpointerParameters CPparams;  
  CPparams.config_prefix = "ckpoint_lat";
  CPparams.rng_prefix = "ckpoint_rng";
  CPparams.saveInterval = 5;
  CPparams.format = "IEEE64BIG";
  
  TheHMC.Resources.LoadNerscCheckpointer(CPparams);

  RNGModuleParameters RNGpar;
  RNGpar.serial_seeds = "1 2 3 4 5";
  RNGpar.parallel_seeds = "6 7 8 9 10";
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  // Construct observables
  typedef PlaquetteMod<HMCWrapper::ImplPolicy> PlaqObs;
  TheHMC.Resources.AddObservable<PlaqObs>();
  //////////////////////////////////////////////

  /////////////////////////////////////////////////////////////
  // Collect actions, here use more encapsulation
  // need wrappers of the fermionic classes 
  // that have a complex construction
  // standard
  RealD beta = 2.25 ;
  WilsonGaugeActionR Waction(beta);
    
  auto GridPtr = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();

  // temporarily need a gauge field
  AdjointRepresentation::LatticeField UA(GridPtr);
  TwoIndexSymmetricRepresentation::LatticeField US(GridPtr);

  Real adjoint_mass = -0.1;
  Real symm_mass = -0.5;
  AdjFermionAction AdjFermOp(UA, *GridPtr, *GridRBPtr, adjoint_mass);
  SymmFermionAction SymmFermOp(US, *GridPtr, *GridRBPtr, symm_mass);

  ConjugateGradient<AdjFermionField> CG_adj(1.0e-8, 10000, false);
  ConjugateGradient<SymmFermionField> CG_symm(1.0e-8, 10000, false);

  // Pass two solvers: one for the force computation and one for the action
  TwoFlavourPseudoFermionAction<AdjImplPolicy> Nf2_Adj(AdjFermOp, CG_adj, CG_adj);
  TwoFlavourPseudoFermionAction<SymmImplPolicy> Nf2_Symm(SymmFermOp, CG_symm, CG_symm);

  // Collect actions
  ActionLevel<LatticeGaugeField, TheRepresentations > Level1(1);
  Level1.push_back(&Nf2_Adj);
  Level1.push_back(&Nf2_Symm);


  ActionLevel<LatticeGaugeField, TheRepresentations > Level2(4);
  Level2.push_back(&Waction);

  TheHMC.TheAction.push_back(Level1);
  TheHMC.TheAction.push_back(Level2);

  // HMC parameters are serialisable 
  TheHMC.Parameters.MD.MDsteps = 20;
  TheHMC.Parameters.MD.trajL   = 1.0;

  TheHMC.ReadCommandLine(argc, argv); // these can be parameters from file
  TheHMC.Run();  // no smearing

  Grid_finalize();
#endif
} // main


#else
int main(int argc, char **argv){}
#endif
