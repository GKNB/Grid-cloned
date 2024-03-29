/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./lib/qcd/modules/plaquette.h

Copyright (C) 2017

Author: Guido Cossu <guido.cossu@ed.ac.uk>

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

#pragma once

NAMESPACE_BEGIN(Grid);

// this is only defined for a gauge theory
template <class Impl>
class PlaquetteLogger : public HmcObservable<typename Impl::Field> {
public:
  // here forces the Impl to be of gauge fields
  // if not the compiler will complain
  INHERIT_GIMPL_TYPES(Impl);

  // necessary for HmcObservable compatibility
  typedef typename Impl::Field Field;

  virtual void TrajectoryComplete(int traj,
                                  ConfigurationBase<Field> &SmartConfig,
                                  GridSerialRNG &sRNG,
                                  GridParallelRNG &pRNG)
  {
    std::cout << GridLogMessage << "+++++++++++++++++++"<<std::endl;
    std::cout << GridLogMessage << "Unsmeared plaquette"<<std::endl;
    TrajectoryComplete(traj,SmartConfig.get_U(false),sRNG,pRNG); // Unsmeared observable
    std::cout << GridLogMessage << "Smeared plaquette"<<std::endl;
    TrajectoryComplete(traj,SmartConfig.get_U(true),sRNG,pRNG); // Unsmeared observable
    std::cout << GridLogMessage << "+++++++++++++++++++"<<std::endl;
  };
  void TrajectoryComplete(int traj,
                          Field &U,
                          GridSerialRNG &sRNG,
                          GridParallelRNG &pRNG) {

    RealD plaq = WilsonLoops<Impl>::avgPlaquette(U);

    int def_prec = std::cout.precision();

    std::cout << GridLogMessage
	      << std::setprecision(std::numeric_limits<Real>::digits10 + 1)
	      << "Plaquette: [ " << traj << " ] "<< plaq << std::endl;

    std::cout.precision(def_prec);

  }
};

NAMESPACE_END(Grid);


