#ifndef MECHANICS_H_
#define MECHANICS_H_
#include <deal.II/dofs/dof_handler.h>
#include "functionEvaluations.h"
#include "supplementaryFunctions.h"

//Saint-Venant Kirchhoff constitutive model
template <int dim>
inline double SVK3D(unsigned int i, unsigned int j, unsigned int k, unsigned int l, Point<dim>& quadPoint, double E){
  double nu=0.3;
  double lambda=(E*nu)/((1+nu)*(1-2*nu)), mu=E/(2*(1+nu));
  return lambda*(i==j)*(k==l) + mu*((i==k)*(j==l) + (i==l)*(j==k));
}

template <int dim>
inline double SVK2D(unsigned int i, unsigned int j, Point<dim>& quadPoint, double E){
  double nu=0.3;
  double lambda=(E*nu)/((1+nu)*(1-2*nu)), mu=E/(2*(1+nu));
  if (i==j && i<2) return lambda + 2*mu;
  else if (i==2 && j==2) return mu;
  else if ((i+j)==1) return lambda;
  else return 0.0;
}

template <int dim>
inline double SVK1D(Point<dim>& quadPoint, double E){
  return E;
}

//Mechanics implementation
template <class T, int dim>
  void getDeformationMap(FEValues<dim>& fe_values, unsigned int DOF, Table<1, T>& ULocal, deformationMap<T, dim>& defMap, const unsigned int iteration){
  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;
  //evaluate dx/dX
  Table<3, T> gradU(n_q_points, dim, dim);
  evaluateVectorFunctionGradient<T, dim>(fe_values, DOF, ULocal, gradU);
  
  //Loop over quadrature points
  for (unsigned int q=0; q<n_q_points; ++q){
    Table<2, T > Fq(dim, dim), invFq(dim, dim); T detFq;
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	defMap.F[q][i][j] = Fq[i][j] = (i==j) + gradU[q][i][j]; //F (as double value)
      }
    }
    getInverse<T, dim>(Fq, invFq, detFq); //get inverse(F)
    defMap.detF[q] = detFq;
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	defMap.invF[q][i][j] = invFq[i][j];
      }
    }
    //detF
    /*
    if (defMap.detF[q].val()<=1.0e-15 && iteration==0){
      printf("\n\n\n**************Non positive jacobian detected**************. Value: %12.4e\n", defMap.detF[q].val());
      for (unsigned int i=0; i<dim; ++i){
	for (unsigned int j=0; j<dim; ++j) printf("%12.6e  ", defMap.F[q][i][j].val());
	printf("\n"); exit(-1);
      }
      //throw "Non positive jacobian detected";
      }*/
  }
}

//Mechanics implementation
template <class T, int dim>
  void evaluateStress(const FEValues<dim>& fe_values,const unsigned int DOF, const Table<1, T>& ULocal, Table<2, double>& Fg, Table<3, T>& P, const typename DoFHandler<dim>::active_cell_iterator& cell,const deformationMap<T, dim>& defMap){
  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;
  
  //Loop over quadrature points
  for (unsigned int q=0; q<n_q_points; ++q){
    //get invFg
    Table<2, double> invFg (dim, dim); double detFg;
    getInverse<double, dim>(Fg, invFg, detFg); //get inverse(Fg)
    
    //Fe
    Table<2, Sacado::Fad::DFad<double> > Fe (dim, dim);
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	Fe[i][j]=0.0;
	for (unsigned int k=0; k<dim; ++k){
	  Fe[i][j] += defMap.F[q][i][k]*invFg[k][j];
	}
      }
    }
    //E
    Table<2, Sacado::Fad::DFad<double> > E (dim, dim);
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	E[i][j] = -0.5*(i==j);
	for (unsigned int k=0; k<dim; ++k){
	  E[i][j] += 0.5*Fe[k][i]*Fe[k][j];
	}
      }
    }
    //S
    Table<2, Sacado::Fad::DFad<double> > S (dim, dim);
    MappingQ<dim> test(1); Point<dim> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(q)));
    if(dim==3){
      for (unsigned int i=0; i<dim; ++i){
	for (unsigned int j=0; j<dim; ++j){
	  S[i][j]=0;
	  for (unsigned int k=0; k<dim; ++k){
	    for (unsigned int l=0; l<dim; ++l){
	      S[i][j] += SVK3D<dim>(i, j, k, l, quadPoint, youngsModulus)*E[k][l];
	    }
	  }
	}
      }
    }
    else if(dim==2){ 
      S[0][0]=SVK2D<dim>(0,0, quadPoint, youngsModulus)*E[0][0]+SVK2D<dim>(0,1, quadPoint, youngsModulus)*E[1][1]+SVK2D<dim>(0,2,quadPoint, youngsModulus)*(E[0][1]+E[1][0]);
      S[1][1]=SVK2D<dim>(1,0, quadPoint, youngsModulus)*E[0][0]+SVK2D<dim>(1,1, quadPoint, youngsModulus)*E[1][1]+SVK2D<dim>(1,2, quadPoint, youngsModulus)*(E[0][1]+E[1][0]);
      S[0][1]=S[1][0]=SVK2D<dim>(2,0, quadPoint, youngsModulus)*E[0][0]+SVK2D<dim>(2,1, quadPoint, youngsModulus)*E[1][1]+SVK2D<dim>(2,2, quadPoint, youngsModulus)*(E[0][1]+E[1][0]);
    }
    else if(dim==1){
      S[0][0]=SVK1D<dim>(quadPoint, youngsModulus)*E[0][0];
    }
    else throw "dim not equal to 1/2/3";
    //P
    for (unsigned int i=0; i<dim; ++i){
      for (unsigned int j=0; j<dim; ++j){
	P[q][i][j]=0;
	for (unsigned int k=0; k<dim; ++k){
	  P[q][i][j]+=Fe[i][k]*S[k][j];
	}
      }
    }
  }
}

//Mechanics residual implementation
template <int dim>
void residualForMechanics(FEValues<dim>& fe_values, unsigned int DOF, Table<1, Sacado::Fad::DFad<double> >& ULocal, Table<1, double>& ULocalConv, Table<1, Sacado::Fad::DFad<double> >& R, typename DoFHandler<dim>::active_cell_iterator& cell, deformationMap<Sacado::Fad::DFad<double>, dim>& defMap, double loadFactor){
  unsigned int dofs_per_cell= fe_values.dofs_per_cell;
  unsigned int n_q_points= fe_values.n_quadrature_points;
  //Temporary arrays
  Table<3,Sacado::Fad::DFad<double> > P (n_q_points, dim, dim);
  Table <2, double> Fg (dim, dim);
  Point<dim> normal(cell->center());
  //compute center of the cell
#if flatPlate==true
  Point<dim> theta(1.0,0,0), theta2(0,0,1.0);
#else
  Point<dim> theta(0,0,0), theta2(0,0,0);
  double norm=std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]);
  theta[0]=normal[1]/norm; theta[1]=-normal[0]/norm;  theta[2]=0.0;
#endif
  //Fg2=theta X theta (outer product)
  Table<2,double> Fg2(dim,dim);
  for (unsigned int i = 0; i < dim; i++){
    for (unsigned int j = 0; j < dim; j++){
      Fg2[i][j]= theta[i]*theta[j] + theta2[i]*theta2[j];
    }
  }
  
  //construct F^g tensor
  for (unsigned int i = 0; i < dim; i++){
    for (unsigned int j = 0; j < dim; j++){
      Fg[i][j]=(double)(i==j)+Fg2[i][j]*FgFactor*loadFactor*(std::pow(normal[2]/ZHeight,ZExponent));
    }
  }
  //printf("%12.6e, %12.6e, %12.6e, %12.6e, %12.6e, %12.6e, %12.6e, %12.6e, %12.6e \n", Fg[0][0], Fg[1][1], Fg[2][2], Fg[0][1], Fg[1][0], Fg[2][0], Fg[0][2], Fg[1][2], Fg[2][1]);
  
  //evaluate mechanics
  evaluateStress<Sacado::Fad::DFad<double>, dim>(fe_values, DOF, ULocal, Fg, P, cell, defMap);
  
  //evaluate Residual
  for (unsigned int i=0; i<dofs_per_cell; ++i) {
    R[i] = 0;
    const unsigned int ck = fe_values.get_fe().system_to_component_index(i).first - DOF;
    if (ck>=0 && ck<dim){
      // R = Grad(w)*P
      for (unsigned int q=0; q<n_q_points; ++q){
	for (unsigned int d = 0; d < dim; d++){
	  R[i] +=  fe_values.shape_grad(i, q)[d]*P[q][ck][d]*fe_values.JxW(q);
	}
      }
    }
  }
}

#endif /* MECHANICS_H_ */
