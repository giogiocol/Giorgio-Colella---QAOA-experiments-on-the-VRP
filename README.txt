Giorgio Colella 
Master Thesis: 'Quantum Computing Potential for Optimization in the Manufacturing Industry'
date: 10/11/2025

The files in this folder are organized as follows.

init_params_impact_on_SVQE_and_QAOA.py file contains the code used to compare the QAOA and S-VQE sensitivity to the choice of the initial parameters values. 

reps_vs_perfor - InitializationStrategy - VRPinstance.py files contain the code used to analyze the algorithms' performances (probability of sampling a feasible solution) varying the number of QAOA repetitions. The same analysis has been conducted with three different initialization strategies (ITERATIVE, NORMALIZED, and RANDOM) and for two VRP instances (SMALL and MEDIUM). All of them used statevector noiseless emulator.

pre_optimization_phase.py file contains the code used for the pre-optimization phase, conducted with noiseless emulator, of the QAOA with reps=2 used for the final experiments on quantum hardware.

QUANTUM_EXPERIMENTS Jupyter notebooks present the code used for the final experiments on quantum hardware for both SMALL and MEDIUM instances.

The VRP_TOY Excel files contain the coordinates of the locations for the two VRP instances (SMALL and MEDIUM) that are required as input for all the other files.