#pragma once
namespace Grid {
template<class Field> class PowerMethod  
{ 
 public: 
  RealD tolerance;  
  int max_iter;
  
  PowerMethod(RealD _tol=0.001, int _max_iter=50): tolerance(_tol), max_iter(_max_iter){ }
  
  template<typename T>  static RealD normalise(T& v) 
  {
    RealD nn = norm2(v);
    nn = sqrt(nn);
    v = v * (1.0/nn);
    return nn;
  }

  RealD operator()(LinearOperatorBase<Field> &HermOp, const Field &src) 
  { 
    GridBase *grid = src.Grid(); 
    
    // quickly get an idea of the largest eigenvalue to more properly normalize the residuum 
    RealD evalMaxApprox = 0.0; 
    auto src_n = src; 
    auto tmp = src; 

    for (int i=0;i< max_iter ;i++) { 
      
      normalise(src_n); 
      HermOp.HermOp(src_n,tmp); 
      RealD vnum = real(innerProduct(src_n,tmp)); // HermOp. 
      RealD vden = norm2(src_n); 
      RealD na = vnum/vden; 

      std::cout << GridLogIterative << "PowerMethod: Current approximation of largest eigenvalue " << na << std::endl;
      
      if ( (fabs(evalMaxApprox/na - 1.0) < tolerance) || (i==max_iter-1) ) { 
 	evalMaxApprox = na; 
	std::cout << GridLogMessage << " Approximation of largest eigenvalue: " << evalMaxApprox << std::endl;
	if( fabs(evalMaxApprox/na - 1.0) > tolerance && i==max_iter-1 )
	  std::cout << GridLogMessage << " Warning: power method did not converge within max iterations " << max_iter << std::endl;	  
	 
 	return evalMaxApprox; 
      } 
      evalMaxApprox = na; 
      src_n = tmp;
    }
    assert(0);
    return 0;
  }
};
}
