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
excel_name = "VRP_TOY_MEDIUM.csv"

# Chiavi delle soluzioni ottimali e fattibili (multi-chiave supportato)
opt_key  = ['001100001011', '100001001101']
feas_key = ['001100001011', '100001001101', '001001010101', '001010001110', '001001100011', '010001001110']

q   = 2
pen = 1

cost_hamiltonian, N = from_problem_exname_to_cost_hamiltonian(excel_name, n_vehicles=q, pen=pen)

num_qubits = cost_hamiltonian.num_qubits
shots = 2 ** num_qubits

# === RANGE DI PROFONDITÀ ===
reps_range = range(1, 7, 1)

# === BEST PARAMS HISTORY (indice 0 -> reps=1, indice 5 -> reps=6) ===
best_params_history = [
    np.array([0.88665575, 5.59345908]),
    np.array([1.62629909, 5.57116878, 4.40770947, 6.27117481]),
    np.array([1.58117791, 6.38731716, 5.25291994, 6.16643497, 2.2638367 , 4.78136535]),
    np.array([3.29164131, 7.67994853, 4.62034276, 7.24198231, 2.10859844, 4.35217443, 7.32036543, 6.16386969]),
    np.array([4.36655302, 7.35511545, 3.99170901, 6.98064144, 1.64329872, 4.0889506 , 7.00803899, 6.15973676, 7.44451573, 3.58191127]),
    np.array([5.3872611 , 7.35930441, 3.99378606, 7.00178928, 1.63510203, 4.09842059, 7.02934344, 7.17423474, 7.44959432, 4.63733954, 2.27007255, 6.02734348])
]

# === CONTATORI / RISULTATI SOLO QAOA (IDEAL / STATEVECTOR) ===
QAOA_gates_counts_ideal = []

# Non facciamo più medie su 50 run: una sola esecuzione per reps
opt_perc_QAOA_ideal  = []  # prob. ottimale per reps
feas_perc_QAOA_ideal = []  # prob. fattibile per reps

for reps in reps_range:
    print(f"-------------- NEW REPS : {reps} ------------------")

    # QAOA ansatz (IDEAL / STATEVECTOR)
    QAOA_circuit = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=reps)

    # Gate counts (senza barrier/measure)
    QAOA_gates_counts_ideal.append(
        sum(1 for instr in QAOA_circuit.data if instr[0].name not in ("barrier", "measure"))
    )

    QAOA_num_params = len(QAOA_circuit.parameters)

    init_params = best_params_history[reps - 1]

    # Controllo coerenza dimensione
    if len(init_params) != QAOA_num_params:
        raise ValueError(
            f"Numero parametri non coerente per reps={reps}: ansatz ne richiede {QAOA_num_params}, "
            f"ma best_params_history ne fornisce {len(init_params)}."
        )

    print(f"QAOA Ideal (depth {reps}) con Best Params preimpostati")

    # Ottimizzazione ideale (statevector) partendo dai parametri forniti
    opt_params, min_cost, opt_time, result = optimize_variational_circuit(
        init_params, QAOA_circuit, cost_hamiltonian, shots,
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

    # Salviamo (una sola volta per reps)
    opt_perc_QAOA_ideal.append(prob_opt)
    feas_perc_QAOA_ideal.append(prob_feas)

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

print("QAOA Ideal Gate Counts:", QAOA_gates_counts_ideal)


'''
# === PLOT SOLO IDEAL ===
import matplotlib.pyplot as plt

def save_line_plot(x, *ys, title="", save_name="plot.png", labels=None, xlabel="x", ylabel="y"):
    plt.figure()
    if labels is None:
        labels = [f"y{i+1}" for i in range(len(ys))]
    for y, label in zip(ys, labels):
        plt.plot(x, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if len(ys) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

# Probabilità soluzione ottima (solo IDEAL)
save_line_plot(
    reps_range, opt_perc_QAOA_ideal,
    title="QAOA (IDEAL) - Optimal Solution Probability vs Reps",
    save_name="QAOA_opt_perc_IDEAL.png",
    labels=["QAOA Ideal"],
    xlabel="Reps", ylabel="Optimal Solution Probability"
)
save_line_plot(
    reps_range, opt_perc_VQE_ideal,
    title="VQE (IDEAL) - Optimal Solution Probability vs Reps",
    save_name="VQE_opt_perc_IDEAL.png",
    labels=["VQE Ideal"],
    xlabel="Reps", ylabel="Optimal Solution Probability"
)

# Gate counts (solo IDEAL, QAOA vs VQE)
save_line_plot(
    reps_range, QAOA_gates_counts_ideal, VQE_gates_counts_ideal,
    title="Gate Counts (IDEAL) vs Reps",
    save_name="Gate_counts_IDEAL.png",
    labels=["QAOA Ideal", "VQE Ideal"],
    xlabel="Reps", ylabel="Number of Gates"
)
'''