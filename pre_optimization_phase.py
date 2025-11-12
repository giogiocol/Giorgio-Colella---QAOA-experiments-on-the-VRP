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

def normalize_linear_quadratic_sparse_pauli_op(hamiltonian: SparsePauliOp) -> SparsePauliOp:
    """
    Normalizza un Hamiltoniano (SparsePauliOp) che contiene solo termini lineari e quadratici
    secondo la formula della Sec. 3.2 del paper Q-2024-01-18-1231.
    """
    # Coefficienti reali
    coeffs = np.real(hamiltonian.coeffs)

    # Etichette Pauli come stringhe ("ZI", "ZZ", ecc.)
    pauli_labels = hamiltonian.paulis.to_labels()

    # Determina qubit totali
    n_qubits = len(pauli_labels[0])
    one_body_coeffs = []
    two_body_coeffs = []

    # Classifica termini 1-body e 2-body
    for label, coeff in zip(pauli_labels, coeffs):
        n_non_identity = n_qubits - label.count("I")
        if n_non_identity == 1:
            one_body_coeffs.append(coeff)
        elif n_non_identity == 2:
            two_body_coeffs.append(coeff)
        # ignora eventuali termini costanti o di ordine superiore

    # Calcola contributi medi
    term1 = np.mean(np.square(one_body_coeffs)) if one_body_coeffs else 0.0
    term2 = np.mean(np.square(two_body_coeffs)) if two_body_coeffs else 0.0

    # Fattore di normalizzazione
    S = np.sqrt(term1 + term2)

    if S == 0:
        return hamiltonian  # evita divisione per zero

    # Hamiltoniana normalizzata
    return SparsePauliOp(hamiltonian.paulis, coeffs=hamiltonian.coeffs / S)

# === PARAMETRI PROBLEMA (invariati) ===
excel_name = "VRP_TOY_SMALL.csv"

feas_key = ['010111']
#feas_key = ['001100001011', '100001001101', '001001010101', '001010001110', '001001100011', '010001001110']

q = 2
pen = 1

cost_hamiltonian, N = from_problem_exname_to_cost_hamiltonian(excel_name, n_vehicles=q, pen=pen)

norm_time = time.time()
cost_hamiltonian = normalize_linear_quadratic_sparse_pauli_op(cost_hamiltonian)
norm_time = time.time() - norm_time

num_qubits = cost_hamiltonian.num_qubits
shots = 2**num_qubits

# === RANGE DI PROFONDITÀ ===
reps_range = range(1, 7, 1)

# === CONTATORI SOLO IDEAL ===
QAOA_gates_counts_ideal = []

opt_perc_QAOA_ideal = []

optimal_params = []

qaoa_params = {
    1: {
        "gamma": [0.5],
        "beta":  [-math.pi/8],           
    },
    2: {
        "gamma": [0.3817, 0.6655],
        "beta":  [-0.4960, -0.2690],
    },
    3: {
        "gamma": [0.3297, 0.5688, 0.6406],
        "beta":  [-0.5500, -0.3675, -0.2109],
    },
    4: {
        "gamma": [0.2949, 0.5144, 0.5586, 0.6429],
        "beta":  [-0.5710, -0.4176, -0.3028, -0.1729],
    },
    5: {
        "gamma": [0.2705, 0.4083, 0.5074, 0.5646, 0.6397],
        "beta":  [-0.5899, -0.4492, -0.3559, -0.2643, -0.1486],
    },
    6: {
        "gamma": [0.2528, 0.4531, 0.4750, 0.5146, 0.5650, 0.6392],
        "beta":  [-0.6004, -0.4670, -0.3880, -0.3176, -0.2325, -0.1291],
    },
    7: {
        "gamma": [0.2383, 0.4327, 0.4516, 0.4830, 0.5147, 0.5686, 0.6393],
        "beta":  [-0.6085, -0.4810, -0.4090, -0.3535, -0.2857, -0.2080, -0.1146],
    },
    8: {
        "gamma": [0.2268, 0.4163, 0.4333, 0.4608, 0.4816, 0.5180, 0.5719, 0.6396],
        "beta":  [-0.6152, -0.4906, -0.4244, -0.3779, -0.3223, -0.2606, -0.1884, -0.1030],
    },
}

reps = 2
# QAOA ansatz (IDEAL / STATEVECTOR)
QAOA_circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)

QAOA_num_params = len(QAOA_circuit.parameters)

init_params = [qaoa_params[reps]['beta'], qaoa_params[reps]['gamma']]
init_params = np.concatenate(init_params)

init_opt_time = time.time()
opt_params, min_cost, opt_time, result = optimize_variational_circuit(
    init_params, QAOA_circuit, cost_hamiltonian, shots,
    backend_noisy=None, mode="statevector"
)

opt_time = time.time() - init_opt_time

optimized_circuit = QAOA_circuit.assign_parameters(opt_params)
optimized_circuit.measure_all()

final_distribution_bin = sample_variational_circuit(
    optimized_circuit, shots, backend=None, mode='statevector'
)

opt_perc_QAOA_ideal.append(sum(final_distribution_bin.get(k, 0.0) for k in feas_key))
optimal_params.append(opt_params)

print('Reps = ', reps)
print('\nFINAL VALUES ----------------------------------------')
print("QAOA Ideal Optimal Solution Percentage:", opt_perc_QAOA_ideal)
print('\nOptimal Params: ', optimal_params)

print('\nNormalization time: ', norm_time)
print('Optimization time:  ', opt_time)
