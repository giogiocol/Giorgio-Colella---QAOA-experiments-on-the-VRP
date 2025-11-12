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


# === PARAMETRI PROBLEMA (invariati) ===
excel_name = "VRP_TOY_SMALL.csv"

feas_key = ['010111']
opt_key = ['010111']
q   = 2
pen = 1

cost_hamiltonian, N = from_problem_exname_to_cost_hamiltonian(excel_name, n_vehicles=q, pen=pen)

num_qubits = cost_hamiltonian.num_qubits
shots = 2 ** num_qubits

# === RANGE DI PROFONDITÀ ===
reps_range = range(1, 7, 1)

# === CONTATORI / RISULTATI SOLO QAOA (IDEAL / STATEVECTOR) ===
QAOA_gates_counts_ideal = []

opt_perc_QAOA_ideal  = []  # media (su 50 run) della prob. ottimale
feas_perc_QAOA_ideal = []  # media (su 50 run) della prob. fattibile

for reps in reps_range:
    print(f"-------------- NEW REPS : {reps} ------------------")

    # QAOA ansatz (IDEAL / STATEVECTOR)
    QAOA_circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)

    # Gate counts (senza barrier/measure)
    QAOA_gates_counts_ideal.append(
        sum(1 for instr in QAOA_circuit.data if instr[0].name not in ("barrier", "measure"))
    )

    QAOA_num_params = len(QAOA_circuit.parameters)

    ### QAOA IDEAL ###
    sampling_values_opt  = []
    sampling_values_feas = []

    for i in range(50):
        print(f"QAOA Ideal (depth {reps}) Run {i+1}/50")
        QAOA_params = np.random.uniform(0, 2 * np.pi, QAOA_num_params)

        # Ottimizzazione ideale (statevector)
        opt_params, min_cost, opt_time, result = optimize_variational_circuit(
            QAOA_params, QAOA_circuit, cost_hamiltonian, shots,
            backend_noisy=None, mode="statevector"
        )

        optimized_circuit = QAOA_circuit.assign_parameters(opt_params)
        optimized_circuit.measure_all()

        final_distribution_bin = sample_variational_circuit(
            optimized_circuit, shots, backend=None, mode='statevector'
        )

        # Probabilità di soluzione ottimale (sommate sulle chiavi ottimali)
        prob_opt = sum(final_distribution_bin.get(k, 0.0) for k in opt_key)

        # Probabilità di soluzione fattibile (sommate sulle chiavi fattibili)
        prob_feas = sum(final_distribution_bin.get(k, 0.0) for k in feas_key)

        sampling_values_opt.append(prob_opt)
        sampling_values_feas.append(prob_feas)

    # Medie su 50 run
    opt_perc_QAOA_ideal.append(np.mean(sampling_values_opt))
    feas_perc_QAOA_ideal.append(np.mean(sampling_values_feas))

    print('VALUES ', reps, 'REPS')
    print("QAOA Ideal Optimal Solution Probability (%): ",
          [round(100 * v, 4) for v in opt_perc_QAOA_ideal])
    print("QAOA Ideal Feasible Solution Probability (%):",
          [round(100 * v, 4) for v in feas_perc_QAOA_ideal])

print('\n\nFINAL VALUES ----------------------------------------')
print("QAOA Ideal Optimal Solution Probability (%): ",
      [round(100 * v, 4) for v in opt_perc_QAOA_ideal])
print("QAOA Ideal Feasible Solution Probability (%):",
      [round(100 * v, 4) for v in feas_perc_QAOA_ideal])