from utils.funzioni import *
from utils.funzioni_adj import *
from utils.funzioni_QAOA import * 
import utils.dicke_states as dk 

import numpy as np
import pandas as pd
import tabulate as tab

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from scipy.optimize import minimize
import time


from qiskit.primitives import StatevectorSampler
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.noise import NoiseModel

import warnings
warnings.filterwarnings("ignore")


#import rustworkx as rx
#from rustworkx.visualization import mpl_draw as draw_graph
#from collections import defaultdict
#from typing import Sequence

def cost_func_estimator_sim(params, ansatz, hamiltonian, estimator):
    global min_cost, opt_params
    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
 
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
 
    results = job.result()[0]
    cost = results.data.evs

    if cost < min_cost:
        min_cost = cost
        opt_params = params

    return cost


def reset_globals():
    global min_cost, opt_params, cost_history, params_history
    min_cost = float('inf')
    opt_params = None
    cost_history = []
    params_history = []

# -------------------------
# Cost function per STATEVECTOR (ideale, analitica)
# -------------------------
def cost_func_estimator_sv(params, ansatz, hamiltonian, estimator):
    """
    Usa StatevectorEstimator (nessuno shot/rumore). Non applica layout all'osservabile.
    """
    global min_cost, opt_params, cost_history, params_history
    pub = (ansatz, hamiltonian, params)
    job = estimator.run([pub])
    res = job.result()
    cost = float(res[0].data.evs)  # EstimatorResult locale

    if cost < min_cost:
        min_cost = cost
        opt_params = np.array(params, dtype=float)

    cost_history.append(cost)
    params_history.append(np.array(params, dtype=float))
    return cost

# -------------------------------------------------------
# Funzione unificata: scegli tra "noisy" e "statevector"
# -------------------------------------------------------
def optimize_variational_circuit(
    params,
    circuit,
    cost_hamiltonian,
    shots,
    backend_noisy,
    *,
    mode="noisy",           # "noisy" (default) oppure "statevector"
    method="COBYLA",
    tol=1e-2
):
    """
    Se mode="noisy":
        - Usa IBM Runtime Estimator dentro Session(backend_noisy)
        - Applica le tue opzioni (DD, twirling, shots)
        - Chiama la tua cost_func_estimator_sim (compatibile con Runtime)

    Se mode="statevector":
        - Usa StatevectorEstimator locale (ideale)
        - Ignora shots/service/backend_noisy
        - Usa cost_func_estimator_sv (nessun layout)
    """
    reset_globals()

    if mode.lower() == "statevector":
        estimator = StatevectorEstimator()
        start_time = time.time()
        result = minimize(
            cost_func_estimator_sv,
            params,
            args=(circuit, cost_hamiltonian, estimator),
            method=method,
            tol=tol,
        )
        opt_time = time.time() - start_time
        return opt_params, min_cost, opt_time, result

    elif mode.lower() == "noisy":
        # Mantiene il tuo flusso originale (IBM Runtime Estimator in sessione)
        start_time = time.time()
        with Session(backend=backend_noisy) as session:
            # NOTE: per qiskit-ibm-runtime < 0.24.0 cambiare mode= in session=
            estimator = Estimator(mode=session)
            estimator.options.default_shots = shots

            # Opzioni di error suppression/mitigation come nel tuo codice
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"

            # ATTENZIONE: qui usiamo la TUA cost_func_estimator_sim (Runtime-style)
            # Deve essere già definita nel tuo ambiente (come nel tuo snippet originale)
            result = minimize(
                cost_func_estimator_sim,
                params,
                args=(circuit, cost_hamiltonian, estimator),
                method=method,
                tol=tol,
            )

        opt_time = time.time() - start_time
        print(result)
        return opt_params, min_cost, opt_time, result

    else:
        raise ValueError("mode deve essere 'noisy' oppure 'statevector'.")


def sample_variational_circuit(
    optimized_circuit,
    shots,
    backend,
    *,
    mode="noisy"   # "noisy" (default) o "statevector"
):
    """
    Esegue il campionamento del circuito variational ottimizzato in modalità:
      - mode="noisy"       → usa IBM Runtime Sampler con mitigazioni
      - mode="statevector" → usa StatevectorSampler locale (ideale, nessun rumore)

    Restituisce:
        final_distribution_bin: dict[str, float]  (probabilità normalizzate)
    """

    if mode.lower() == "statevector":
        # -----------------------
        # IDEALE: STATEVECTOR
        # -----------------------
        sampler = StatevectorSampler()
        pub = (optimized_circuit,)

        job = sampler.run([pub])  # deterministico, nessuno shot
        res = job.result()[0]

        probs = res.data.meas.get_counts()
        tot_count = sum(probs.values())
        final_distribution_bin = {key: val / tot_count for key, val in probs.items()}
        return final_distribution_bin

    elif mode.lower() == "noisy":
        # -----------------------
        # NOISY: IBM RUNTIME SAMPLER
        # -----------------------
        with Session(backend=backend) as session:
            sampler = Sampler(mode=session)
            sampler.options.default_shots = shots

            # Simple error suppression / mitigation
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.num_randomizations = "auto"

            pub = (optimized_circuit,)
            job = sampler.run([pub], shots=int(shots))
            res = job.result()[0]
            counts_bin = res.data.meas.get_counts()

        final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
        return final_distribution_bin

    else:
        raise ValueError("mode deve essere 'noisy' oppure 'statevector'.")
    

### MAIN CODE ###

excel_name = "VRP_TOY_SMALL.csv"
opt_key = '010111'

q = 2
pen = 1
reps = 2

runs = 1000


cost_hamiltonian, N = from_problem_exname_to_cost_hamiltonian(excel_name, n_vehicles=q, pen=pen)

num_qubits = cost_hamiltonian.num_qubits
shots = 2**num_qubits

QAOA_circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)
QAOA_num_params = len(QAOA_circuit.parameters)

VQE_circuit = torino_native_ansatz(num_qubits, reps)
VQE_num_params = len(VQE_circuit.parameters)

qaoa_results = []
vqe_results = []

for i in range(runs):
    if i % 100 == 0:
        print(f"Processing... {(i/runs)*100}%")
    np.random.seed(i)

    # QAOA IDEAL
    QAOA_params = np.random.uniform(0, 2 * np.pi, QAOA_num_params)
    opt_params, min_cost, opt_time, result = optimize_variational_circuit(QAOA_params, QAOA_circuit, 
                                                                          cost_hamiltonian, shots, backend_noisy=None, mode="statevector")
    optimized_circuit = QAOA_circuit.assign_parameters(opt_params)
    optimized_circuit.measure_all()

    final_distribution_bin = sample_variational_circuit(optimized_circuit, shots, backend=None, mode='statevector')

    qaoa_results.append(round(final_distribution_bin[opt_key], 4) if opt_key in final_distribution_bin else 0)

    # VQE IDEAL
    VQE_params = np.random.uniform(0, 2 * np.pi, VQE_num_params)
    opt_params, min_cost, opt_time, result = optimize_variational_circuit(VQE_params, VQE_circuit,
                                                                          cost_hamiltonian, shots, backend_noisy=None, mode="statevector")
    optimized_circuit = VQE_circuit.assign_parameters(opt_params)
    optimized_circuit.measure_all() 

    final_distribution_bin = sample_variational_circuit(optimized_circuit, shots, backend=None, mode='statevector')
    vqe_results.append(round(final_distribution_bin[opt_key], 4) if opt_key in final_distribution_bin else 0)


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 5))
plt.hist(qaoa_results, bins=20, weights=np.ones_like(qaoa_results)/len(qaoa_results),
         alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Optimal Solution Probability')
plt.ylabel('Probability')
plt.title('Optimal Solution Probability Distribution - QAOA')
plt.xlim(0, 1)
plt.savefig('QAOA_prob_dist_varying_init_params.png')

print(f"QAOA Range: {min(qaoa_results)} - {max(qaoa_results)}")
print(f"QAOA Standard deviation: {np.std(qaoa_results)}")


plt.figure(figsize=(8, 5))
plt.hist(vqe_results, bins=20, weights=np.ones_like(vqe_results)/len(vqe_results),
         alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Optimal Solution Probability')   
plt.ylabel('Probability')
plt.title('Optimal Solution Probability Distribution - S-VQA')
plt.xlim(0, 1)
plt.savefig('VQE_prob_dist_varying_init_params.png')

print(f"S-VQA Range: {min(vqe_results)} - {max(vqe_results)}")
print(f"S-VQA Standard deviation: {np.std(vqe_results)}")


plt.figure(figsize=(16, 5))
plt.hist(qaoa_results, bins=20, weights=np.ones_like(qaoa_results)/len(qaoa_results),
         alpha=0.7, color='blue', edgecolor='black', label='QAOA')
plt.hist(vqe_results, bins=20, weights=np.ones_like(vqe_results)/len(vqe_results),
         alpha=0.7, color='green', edgecolor='black', label='S-VQA')
plt.xlabel('Optimal Solution Probability')   
plt.ylabel('Probability')
plt.yscale('log')
plt.legend()
plt.title('Optimal Solution Probability Distribution')
plt.xlim(0, 1)
plt.savefig('TOTAL_prob_dist_varying_init_params.png')




