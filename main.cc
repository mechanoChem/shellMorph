//include headers
#include  <deal.II/base/logstream.h>
#include  <deal.II/base/quadrature_lib.h>
#include  <deal.II/base/function.h>
#include  <deal.II/lac/block_vector.h>
#include  <deal.II/lac/full_matrix.h>
#include  <deal.II/lac/block_sparse_matrix.h>
#include  <deal.II/lac/sparse_direct.h>
#include  <deal.II/lac/precondition.h>
#include  <deal.II/lac/solver_cg.h>
#include  <deal.II/lac/solver_bicgstab.h>
#include  <deal.II/lac/solver_gmres.h>
#include  <deal.II/lac/constraint_matrix.h>
#include  <deal.II/grid/tria.h>
#include  <deal.II/grid/grid_generator.h>
#include  <deal.II/grid/persistent_tria.h>
#include  <deal.II/grid/intergrid_map.h>
#include  <deal.II/grid/grid_in.h>
#include  <deal.II/grid/tria_accessor.h>
#include  <deal.II/grid/tria_iterator.h>
#include  <deal.II/grid/grid_tools.h>
#include  <deal.II/grid/grid_refinement.h>
#include  <deal.II/dofs/dof_handler.h>
#include  <deal.II/dofs/dof_renumbering.h>
#include  <deal.II/dofs/dof_accessor.h>
#include  <deal.II/dofs/dof_tools.h>
#include  <deal.II/fe/mapping_q1.h>
#include  <deal.II/fe/fe_raviart_thomas.h>
#include  <deal.II/fe/fe_dgq.h>
#include  <deal.II/fe/fe_system.h>
#include  <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include  <deal.II/numerics/data_out.h>
#include  <deal.II/numerics/error_estimator.h>
#include  <deal.II/numerics/solution_transfer.h>
#include  <deal.II/base/tensor_function.h>
#include  <deal.II/base/thread_management.h>
#include  <deal.II/base/multithread_info.h>
#include  <tbb/task_scheduler_init.h>
#include  <Sacado.hpp>
#include  <stdio.h>
#include  <stdlib.h>
#include  <iostream>
#include  <fstream>
#include  <sstream>
#include  <math.h>
#include  <string>
#include  <time.h>
#include "functionEvaluations.h"
using namespace dealii;

//geometry
#define flatPlate false
//
#if flatPlate==true
//flat
#define flatX 0.1
#define flatY 0.002
#define slicesAlongHeight 15
#define ZHeight 0.066
#define refineLevels 0
//
#else
//shell
#define innerRadius 1.0
#define shellWidth 0.01
#define slicesAlongHeight 10
#define ZHeight 0.1
#define refineLevels 0
#endif
//
#define youngsModulus 1.0
#define FgFactor 0.05
#define ZExponent 0.0
//
#include "mechanics.h"
//
#define numThreads 1
#define dims 3
#define degrees 1

//time stepping
#define numTimeSteps 10
#define machineEPS 1.0e-14
#define frontPropagations 25

//
#if flatPlate==true
template <int dim>
Point<dim> warp (const Point<dim> &p)
{
  Point<dim> q = p;
  //double value=0.0;
  //if ( (q[2]>0) && (q[2]<flatHeight) && (q[0]>0) && (q[0]<flatX) ){
  //  value=((double)(std::rand()%10))/9.0-0.5;
  //}
  
  double c=q[0]/flatX;
  q[1]+= 4*c*(1-c)*flatY/4.0;
  return q;
}
#endif

template <int dim, int degree>
class seaShellProblem
{
public:
  seaShellProblem (unsigned int _temp);
  ~seaShellProblem();
  void run ();
  void assemble_system (int iteration);
  void assemble_system_interval (int iteration, const typename DoFHandler<dim>::active_cell_iterator &begin, const typename DoFHandler<dim>::active_cell_iterator &end);
  void solve (unsigned int timestep_number);
  void output_results (const unsigned int cycle, bool frontFile=false);
  void traceBoundaryNodes();
  
  FESystem<dim>        fe;
  Triangulation<dim>   triangulation;
  DoFHandler<dim>      dof_handler;
  QGauss<dim> 	       quadrature_formula;
  
  //Data structures related to the system of equations
  SparseMatrix<double> system_matrix, system_matrix2;
  Vector<double>       system_rhs, U, Un, dU, UOld;

  //Data structures related to the mesh
  SparsityPattern     sparsity_pattern;
  SparsityPattern     sparsity_pattern2;
  ConstraintMatrix    constraints, constraintsZero;
  ConstraintMatrix    constraints2, constraints2Zero;
  std::map<unsigned int, unsigned int> mapZ;
  
  //Solution variables
  unsigned int timestep_number;
  double loadFactor; unsigned int frontID;
  //Output
  std::string outputFilename;
  std::vector<std::string> nodal_solution_names; std::vector<DataComponentInterpretation::DataComponentInterpretation> nodal_data_component_interpretation, cell_data_component_interpretation;
  DataOut<dim> data_outComposite;
  //Other
  Threads::Mutex    assembler_lock;
};

//Class constructor
template <int dim, int degree>
seaShellProblem<dim, degree>::seaShellProblem (unsigned int _temp):
  fe (FE_Q<dim>(degree), dim),
  dof_handler (triangulation),
  quadrature_formula(degree+2){
  std::srand (2);
  //variables
  MultithreadInfo::set_thread_limit(numThreads);
  //solution control
  timestep_number=1; loadFactor=0.0; frontID=0;
  outputFilename="output";
  //Nodal Solution names
  for (unsigned int i=0; i<dim; ++i){
    nodal_solution_names.push_back("u"); nodal_data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  }
}

template <int dim, int degree>
seaShellProblem<dim, degree>::~seaShellProblem (){dof_handler.clear ();}


template <int dim, int degree>
void seaShellProblem<dim, degree>::assemble_system(int iteration){
  system_matrix=0; system_matrix2=0;
  system_rhs=0; 

  //start threads
  const unsigned int n_threads=numThreads;
  Threads::ThreadGroup<> threads;
  typedef typename DoFHandler<dim>::active_cell_iterator active_cell_iterator;
  std::vector<std::pair<active_cell_iterator,active_cell_iterator> > thread_ranges = Threads::split_range<active_cell_iterator> (dof_handler.begin_active (), dof_handler.end (), n_threads);
  for (unsigned int thread=0; thread<n_threads; ++thread){
    threads += Threads::new_thread (&seaShellProblem<dim, degree>::assemble_system_interval, *this, iteration, thread_ranges[thread].first, thread_ranges[thread].second);
  }
  threads.join_all ();
}

//Assembly
template <int dim, int degree>
void seaShellProblem<dim, degree>::assemble_system_interval (int iteration, const typename DoFHandler<dim>::active_cell_iterator &begin, const typename DoFHandler<dim>::active_cell_iterator &end){
  //Initialize data structures
  FEValues<dim> fe_values (fe, quadrature_formula, update_values   | update_gradients | update_quadrature_points | update_JxW_values);
  
  //Temporary variables
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   num_quad_points    = quadrature_formula.size();
  FullMatrix<double>   local_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       local_rhs (dofs_per_cell);
  std::vector<unsigned int> local_dof_indices (dofs_per_cell);
  
  //Loop over elements for assembly
  typename DoFHandler<dim>::active_cell_iterator cell = begin, endc = end;
  for (;cell!=endc; ++cell){
    fe_values.reinit (cell);
    local_matrix = 0; local_rhs = 0;
    cell->get_dof_indices (local_dof_indices);
    
    //AD variables
    Table<1, Sacado::Fad::DFad<double> > ULocal(dofs_per_cell); Table<1, double > ULocalConv(dofs_per_cell);
    for (unsigned int i=0; i<dofs_per_cell; ++i){
      ULocal[i]= U(local_dof_indices[i]); ULocal[i].diff (i, dofs_per_cell);
      ULocalConv[i]= Un(local_dof_indices[i]);
    }
    deformationMap<Sacado::Fad::DFad<double>, dim> defMap(num_quad_points);
    
    getDeformationMap<Sacado::Fad::DFad<double>, dim>(fe_values, 0, ULocal, defMap, iteration);
    //evaluate Mechanics fields
    Table<1, Sacado::Fad::DFad<double> > mechanicsRes(dofs_per_cell);
    residualForMechanics<dim>(fe_values, 0, ULocal, ULocalConv, mechanicsRes, cell, defMap, loadFactor);

    //Residual(R) and Jacobian(R')
    for (unsigned int i=0; i<dofs_per_cell; ++i) {
      Sacado::Fad::DFad<double> R =mechanicsRes[i];
      for (unsigned int j=0; j<dofs_per_cell; ++j){
	// R' by AD
	local_matrix(i,j)= R.fastAccessDx(j);
      }
      //R
      local_rhs(i) -= R.val();
    }
    
    //Global Assembly
    assembler_lock.acquire ();
    if (frontID==1){
      constraintsZero.distribute_local_to_global (local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs);
    }
    else{
      constraints2Zero.distribute_local_to_global (local_matrix, local_rhs, local_dof_indices, system_matrix2, system_rhs);
    }
    assembler_lock.release ();
  }
}


//Nonlinear solve
template <int dim, int degree>
void seaShellProblem<dim, degree>::solve(unsigned int timestep_number){
  unsigned int iter=0;
  double initNorm;
  while (true){
    if (iter>100){printf ("Maximum number of iterations reached without convergence. \n"); break;}
    //Compute residual forces and norms
    assemble_system(iter);
    double norm=system_rhs.l2_norm();
    if (iter==0) {initNorm=norm;}
    double relNorm=norm/initNorm;
    //Output convergence data
    printf ("Increment:%u, iteration: %u, rhs norm: %8.2e, relative norm: %8.2e, load: %8.2e\n", timestep_number, iter, norm, relNorm, loadFactor);
    if ((relNorm<1.0e-10) || (norm<machineEPS)){break;}

    //Solving
    dU=0;
    SparseDirectUMFPACK  A_direct;
    if (frontID==1){
      A_direct.initialize(system_matrix);
    }
    else{
      A_direct.initialize(system_matrix2);
    }
    A_direct.vmult (dU, system_rhs);
    printf("direct solve complete.\n");
    if(frontID==1){
      constraintsZero.distribute (dU);
    }
    else{
      constraints2Zero.distribute (dU);
    }
    U+=dU; ++iter;
  }
  Un=U;
}

//Output results
template <int dim, int degree>
void seaShellProblem<dim, degree>::output_results (const unsigned int cycle, bool frontFile){
  //Write results to VTK file
  char filename [50]; sprintf (filename, "%s-%u.vtk", outputFilename.c_str(), cycle); std::ofstream output1 (filename);
  DataOut<dim> data_out; data_out.attach_dof_handler (dof_handler);
  //Add nodal DOF data
  data_out.add_data_vector (U, nodal_solution_names, DataOut<dim>::type_dof_data, nodal_data_component_interpretation);
#if flatPlate==true
  data_out.build_patches (degree+1);
#else
    data_out.build_patches ();
#endif
  data_out.write_vtk (output1); output1.close();
  printf ("output written to files: %s\n", filename);
  //
  if (frontFile){
    if (frontID==1){
      data_outComposite.attach_dof_handler (dof_handler);
      //Add nodal DOF data
      data_outComposite.add_data_vector (U, nodal_solution_names, DataOut<dim>::type_dof_data, nodal_data_component_interpretation);
#if flatPlate==true
      data_outComposite.build_patches (degree+1);
#else
      data_outComposite.build_patches ();
#endif
    }
    else{
#if flatPlate==true
      data_outComposite.merge_patches(data_out, Point<dim>(0,0,(frontID-1)*ZHeight));
#else
      data_outComposite.merge_patches(data_out, Point<dim>(0,0,(frontID-1)*ZHeight));
#endif
    }
    sprintf (filename, "front-%s-%u.vtk", outputFilename.c_str(), frontID); std::ofstream output2 (filename);
    data_outComposite.write_vtk (output2); output2.close();
    printf ("output written to files: %s\n", filename);
  }
}


//Problem setup and time stepping
template <int dim, int degree>
void seaShellProblem<dim, degree>::traceBoundaryNodes(){
  std::cout << "tracing boundary-Z\n";
  MappingQ<dim, dim> mapping(degree, true);
  std::vector<Point<dim> > support_points(dof_handler.n_dofs(), Point<dim>(0,0,0));
  DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);
  for (unsigned int k=0; k<dim; k++){
    std::vector<bool> mask(dim, false); mask[k]=true;
    std::set<types::boundary_id> boundary_ids; boundary_ids.insert(33); boundary_ids.insert(44);
    std::vector <bool> selected_dofs(dof_handler.n_dofs(), false);
    DoFTools::extract_boundary_dofs(dof_handler, ComponentMask(mask), selected_dofs, boundary_ids);
  
    //loop over selected DOFS
    for (unsigned int i=0; i<selected_dofs.size(); i++){
      if (selected_dofs[i]==true){
	if  (std::abs(support_points[i][2])<1.0e-10){
	  bool found=false;
	  for (unsigned int j=0; j<selected_dofs.size(); j++){
	     if (selected_dofs[j]==true){
	       if (std::abs(std::abs(support_points[j][2]-support_points[i][2])-ZHeight)<1.0e-10){
		 if ((std::abs(support_points[i][0]-support_points[j][0])<1.0e-12) && (std::abs(support_points[i][1]-support_points[j][1])<1.0e-12)){
		   mapZ[i]=j; found=true;
		   //printf("i: %u, x: %12.4e, y:%12.4e, z:%12.4e, j:%u, xz: %12.4e, yz:%12.4e, zz:%12.4e\n", i, support_points[i][0], support_points[i][1], support_points[i][2], j, support_points[j][0], support_points[j][1], support_points[j][2]);
		   break;
		 }
	       }
	    }
	  }
	  if (!found){std::cout<< "boundary Z not found\n"; exit(-1);}
	}
      }
    }
  }
  std::cout << "boundary-Z size: " << mapZ.size() << "\n";
}

//Problem setup and time stepping
template <int dim, int degree>
void seaShellProblem<dim, degree>::run(){
  Triangulation<2> mesh2D ;
#if flatPlate==true
  std::vector< unsigned int > repetitions; repetitions.push_back(((unsigned int) (flatX/flatY))/2); repetitions.push_back(2);
  GridGenerator::subdivided_hyper_rectangle(mesh2D, repetitions, Point<2>(0,0), Point<2>(flatX, flatY), false);
  //GridTools::rotate(3.142/10, mesh2D);
   //extrude
  GridGenerator::extrude_triangulation (mesh2D, slicesAlongHeight, ZHeight, triangulation);
  GridTools::transform (&warp<dim>, triangulation);   
#else
  GridGenerator::quarter_hyper_shell(mesh2D, Point<2>(), innerRadius, innerRadius+shellWidth,0,false);
  //extrude
  GridGenerator::extrude_triangulation (mesh2D, slicesAlongHeight, ZHeight, triangulation);
#endif
  //refine
  triangulation.refine_global (refineLevels);
  //DOF's
  dof_handler.distribute_dofs (fe);
  DoFRenumbering::Cuthill_McKee (dof_handler); //reordering DOF for minimizing bandwidth
  //write mesh out to file
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.build_patches ();
  std::ofstream output ("mesh.vtk");
  data_out.write_vtk (output);

  //mark boundaries
  unsigned int countX=0, countY=0, countZ=0;
  typename Triangulation<dim>::cell_iterator
    cell = triangulation.begin (),
    endc = triangulation.end();
  for (; cell!=endc; ++cell){
    for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face){
      if (cell->face(face)->at_boundary()){
#if flatPlate==true
	//X=0
	if (std::abs(cell->face(face)->center()[0])<1.0e-10) {
	  cell->face(face)->set_boundary_id (11); countX++;}
	else if (std::abs(cell->face(face)->center()[0]-flatX)<1.0e-10){
	  cell->face(face)->set_boundary_id (22); countY++;}
	else if (std::abs(cell->face(face)->center()[2])<1.0e-10){
	  cell->face(face)->set_boundary_id (33); countZ++;}
	else if (std::abs(cell->face(face)->center()[2]-ZHeight)<1.0e-10){
	  cell->face(face)->set_boundary_id (44);}
#else
	//X=0
	if (std::abs(cell->face(face)->center()[0])<1.0e-10) {
	  cell->face(face)->set_boundary_id (11); countX++;}
	else if (std::abs(cell->face(face)->center()[1])<1.0e-10){
	  cell->face(face)->set_boundary_id (22); countY++;}
	else if (std::abs(cell->face(face)->center()[2])<1.0e-10){
	  cell->face(face)->set_boundary_id (33); countZ++;}
	else if (std::abs(cell->face(face)->center()[2]-ZHeight)<1.0e-10){
	  cell->face(face)->set_boundary_id (44);}
#endif
      }
    }
  }
  std::cout << "Dirichlet faces: " << "X:" << countX  << " Y:" << countY  << " Z:" << countZ << std::endl;
  //constraints
  //XYZ
#if flatPlate==true
  std::vector<bool> maskXY(dim, true); maskXY[2]=false;
  std::vector<bool> maskXYZ(dim, true);
  DoFTools::make_zero_boundary_constraints (dof_handler, 11, constraints, ComponentMask (maskXYZ));
  DoFTools::make_zero_boundary_constraints (dof_handler, 22, constraints, ComponentMask (maskXYZ));
  DoFTools::make_zero_boundary_constraints (dof_handler, 33, constraints, ComponentMask (maskXYZ));
  //
  DoFTools::make_zero_boundary_constraints (dof_handler, 11, constraintsZero, ComponentMask (maskXYZ));
  DoFTools::make_zero_boundary_constraints (dof_handler, 22, constraintsZero, ComponentMask (maskXYZ));
  DoFTools::make_zero_boundary_constraints (dof_handler, 33, constraintsZero, ComponentMask (maskXYZ));
#else
  std::vector<bool> maskXYZ(dim, true), maskX(dim, false), maskY(dim, false); maskX[0]=true; maskY[1]=true;
  DoFTools::make_zero_boundary_constraints (dof_handler, 33, constraints, ComponentMask (maskXYZ));
  DoFTools::make_zero_boundary_constraints (dof_handler, 11, constraints, ComponentMask (maskXYZ));
  DoFTools::make_zero_boundary_constraints (dof_handler, 22, constraints, ComponentMask (maskXYZ));
  //
  DoFTools::make_zero_boundary_constraints (dof_handler, 33, constraintsZero, ComponentMask (maskXYZ));
  DoFTools::make_zero_boundary_constraints (dof_handler, 11, constraintsZero, ComponentMask (maskXYZ));
  DoFTools::make_zero_boundary_constraints (dof_handler, 22, constraintsZero, ComponentMask (maskXYZ));
  //
  DoFTools::make_zero_boundary_constraints (dof_handler, 33, constraints2, ComponentMask (maskXYZ));
  DoFTools::make_zero_boundary_constraints (dof_handler, 11, constraints2, ComponentMask (maskX));
  DoFTools::make_zero_boundary_constraints (dof_handler, 22, constraints2, ComponentMask (maskY));
  //
  DoFTools::make_zero_boundary_constraints (dof_handler, 33, constraints2Zero, ComponentMask (maskXYZ));
  DoFTools::make_zero_boundary_constraints (dof_handler, 11, constraints2Zero, ComponentMask (maskX));
  DoFTools::make_zero_boundary_constraints (dof_handler, 22, constraints2Zero, ComponentMask (maskY));
#endif
  std::cout <<  "constraintsXYZ: " << constraints.n_constraints() << std::endl;
  //
  constraints.close (); constraintsZero.close();
  constraints2.close (); constraints2Zero.close();
  //
  traceBoundaryNodes();
  //
  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DynamicSparsityPattern dsp2(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  DoFTools::make_sparsity_pattern (dof_handler, dsp2);
  constraints.condense (dsp);
  constraints2.condense (dsp2);
  sparsity_pattern.copy_from (dsp);
  sparsity_pattern2.copy_from (dsp2);
  
  //Initialize data structures
  system_matrix.reinit (sparsity_pattern);
  system_matrix2.reinit (sparsity_pattern2);
  U.reinit (dof_handler.n_dofs()); Un.reinit (dof_handler.n_dofs());  dU.reinit (dof_handler.n_dofs()); UOld.reinit (dof_handler.n_dofs()); system_rhs.reinit (dof_handler.n_dofs());
    
  //Initial conditions
  Un=0.0; U=Un; UOld=0.0;

  //output Cell, DOF information
  std::cout << "Number of active cells:       " << triangulation.n_active_cells() << std::endl;
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
  
  //Time stepping
  unsigned int outputCount=0;
  output_results(outputCount++);
  for (unsigned int i=0; i<frontPropagations; i++){
    frontID=i+1;
    printf ("\nPropagating Front:%u \n", i+1);
    //
    if (i>0){
      //printf("before constraints size: %u\n", constraints.n_constraints());
      for (std::map<unsigned int, unsigned int>::iterator it=mapZ.begin(); it!=mapZ.end(); ++it){
	constraints2.set_inhomogeneity(it->first, UOld(it->second));
	//std::cout << it->first  << ": " << UOld(it->second) << " ... ";
      }
      constraints2.distribute(U);
      Un=U;
      //printf("\nafter constraints size: %u\n", constraints.n_constraints());
    }
    //
    timestep_number=0;
    while (timestep_number <= numTimeSteps){
      printf ("\nTimestep:%u \n", timestep_number);
      loadFactor=((double)timestep_number)/numTimeSteps;
      //Solve
      solve(timestep_number);
      if (loadFactor==1.0){output_results(outputCount++, true);}
      else{output_results(outputCount++);}
      ++timestep_number;
    }
    //transfer U values on Boundary 44 to BC's on Boundary 33
    UOld=Un; U=0; Un=0;
  }
}

//main
int main (){
  try{
    seaShellProblem<dims, degrees> problemObject(0);
    problemObject.run();
  }
  catch (std::exception &exc){
    std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl << exc.what() << std::endl << "Aborting!" << std::endl;
  }
  catch (char * str ){
    std::cout << "Exception raised: " << str << std::endl;
  }
  catch (...){
    std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl;
  }
  return 0;
}


