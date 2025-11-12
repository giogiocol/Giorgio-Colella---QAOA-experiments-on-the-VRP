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


excel_name = "VRP_TOY_SMALL.csv"

opt_key = feas_key = ['010111']
#opt_key = ['001100001011', '100001001101']
#feas_key =  ['001100001011', '100001001101', '001001010101', '001010001110', '001001100011', '010001001110']

q = 2
pen = 1

# -------------------------
# Global runtime bookkeeping
# -------------------------
global_t0 = time.time()

# -------------------------
# Build cost Hamiltonian
# -------------------------
cost_hamiltonian, N = from_problem_exname_to_cost_hamiltonian(excel_name, n_vehicles=q, pen=pen)
num_qubits = cost_hamiltonian.num_qubits
shots = 2**num_qubits


# -------------------------
# Common grid for beta/gamma sweeps
# -------------------------
params_range = []
for i in range(1, 11):
    params_range.append((i * 2 * np.pi) / 10)


def qaoa_optimize_until_reps(
    desired_reps: int,
    cost_hamiltonian,
    params_range,
    shots: int,
    opt_key: str,
    optimize_variational_circuit,
    sample_variational_circuit,
    backend_noisy=None,
    mode: str = "statevector",
):
    """
    Esegue QAOA iterativamente da p=1 fino a desired_reps.
    Per p=1 fa uno sweep su (beta1, gamma1) e ottimizza.
    Per p>1 blocca i 2(p-1) migliori precedenti e fa sweep su (beta_p, gamma_p),
    usando ciascuna coppia come seed per ottimizzare il vettore completo di 2p parametri.

    Stampa:
      - progresso per ogni reps
      - percentuale ottima parziale a ogni step
      - percentuale ottima finale
      - tempo totale

    Ritorna:
      history = {
          "reps": [1, 2, ..., desired_reps],
          "best_params": [list_params_p1, list_params_p2, ...],
          "best_prob": [prob_p1, prob_p2, ...],
          "time": [t1, t2, ...]
      }
    """
    if desired_reps < 1:
        raise ValueError("desired_reps deve essere >= 1")

    global_start = time.time()

    # --- Storico ---
    history = {
        "reps": [],
        "best_params": [],
        "best_prob_opt": [],
        "best_prob_feas": [],
        "time": []
    }

    best_params_prev = []
    best_prob_prev_opt = -1.0
    best_prob_prev_feas = -1.0

    for p in range(1, desired_reps + 1):
        print(f"\n=== Ottimizzazione QAOA (reps = {p}) ===")
        phase_t0 = time.time()

        # Ansatz QAOA con p layer
        qaoa_circ = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=p)
        current_best = {"params": None, "prob_opt": -1.0, "prob_feas": -1.0}

        total_sweep = len(params_range) ** 2
        sweep_counter = 0

        # Sweep su (beta_p, gamma_p)
        for beta in params_range:
            for gamma in params_range:
                sweep_counter += 1
                print(f"  [reps={p}] Sweep {sweep_counter}/{total_sweep}...", end="\r")

                if p == 1:
                    init_params = [beta, gamma]
                else:
                    init_params = [0] * (2 * p)
                    for i in range(p-1):
                        init_params[i] = best_params_prev[i]
                        init_params[i + p] = best_params_prev[i + (p - 1)]
                    init_params[p - 1] = beta
                    init_params[2 * p - 1] = gamma

                # Ottimizza tutti i 2p parametri
                opt_params, min_cost, opt_time, _ = optimize_variational_circuit(
                    init_params,
                    qaoa_circ,
                    cost_hamiltonian,
                    shots,
                    backend_noisy=backend_noisy,
                    mode=mode
                )

                # Calcolo probabilità dell'optimum
                optimized_circuit = qaoa_circ.assign_parameters(opt_params)
                optimized_circuit.measure_all()
                dist = sample_variational_circuit(
                    optimized_circuit,
                    shots,
                    backend=None,
                    mode=mode
                )

                prob_opt = sum(dist.get(k, 0.0) for k in opt_key)
                prob_feas = sum(dist.get(k, 0.0) for k in feas_key)
              
                if prob_feas > current_best["prob_feas"]:
                    current_best["prob_feas"] = prob_feas
                    current_best["params"] = opt_params
                    current_best["prob_opt"] = prob_opt

        # Fine della fase p
        phase_time = time.time() - phase_t0
        best_params_prev = current_best["params"]
        best_prob_prev_opt = current_best["prob_opt"]
        best_prob_prev_feas = current_best["prob_feas"]

        print(f"\n[reps={p}] → Best optimal prob: {100 * best_prob_prev_opt:.4f}% | Time: {phase_time:.2f}s")
        print(f"[reps={p}] → Best feasible prob: {100 * best_prob_prev_feas:.4f}% | Time: {phase_time:.2f}s")

        # Salvataggio storico
        history["reps"].append(p)
        history["best_params"].append(best_params_prev)
        history["best_prob_opt"].append(best_prob_prev_opt)
        history["best_prob_feas"].append(best_prob_prev_feas)
        history["time"].append(phase_time)

    total_time = time.time() - global_start
    print(f"\n=== Risultati finali ===")
    print(f"Reps massimi: {desired_reps}")
    print(f"Percentuale feasible finale: {100 * best_prob_prev_feas:.4f}%")
    print(f"Percentuale ottima finale: {100 * best_prob_prev_opt:.4f}%")
    print(f"Tempo totale di ottimizzazione: {total_time:.2f} s")

    return history
# =========================

result = qaoa_optimize_until_reps(
    desired_reps=6,
    cost_hamiltonian=cost_hamiltonian,
    params_range=params_range,
    shots=shots,
    opt_key=opt_key,
    optimize_variational_circuit=optimize_variational_circuit,
    sample_variational_circuit=sample_variational_circuit,
    backend_noisy=None,
    mode="statevector"  
)

# =========================

print("\n=== Final Results ===")

print("Reps history:", result["reps"])
print("Best params history:", result["best_params"])
print("Best opt_prob history:", result["best_prob_opt"])
print("Best feas_prob history:", result["best_prob_feas"])
print("Time history:", result["time"])

