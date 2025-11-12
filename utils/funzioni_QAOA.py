import pandas as pd
import numpy as np
import tabulate as tab
import random
import networkx as nx
import matplotlib.pyplot as plt
import csv
import math


import numpy as np
import pandas as pd
import tabulate as tab

from qiskit_optimization import QuadraticProgram
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
 
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QAOAAnsatz

#from qiskit_ibm_provider import IBMProvider
from qiskit.providers.jobstatus import JobStatus
import time

import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
from scipy.optimize import minimize
from collections import defaultdict
from typing import Sequence

def from_qubo_to_qp(Q_qubo):
    qp = QuadraticProgram()

    n= Q_qubo.shape[0]

    for i in range(n):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": Q_qubo[i, i] for i in range(n)}

    quadratic = {}
    for i in range(n):
        for j in range(i+1, n):
            coeff = Q_qubo[i, j] + Q_qubo[j, i]
            if coeff != 0:
                quadratic[(f"x{i}", f"x{j}")] = coeff

    qp.minimize(linear=linear, quadratic=quadratic)

    return qp

import pandas as pd
from scipy.optimize import minimize
from qiskit_ibm_runtime import Estimator, Session

# Scrivi header all'inizio del file
def initialize_excel(num_params, excel_filename):
    columns = [f"param_{i}" for i in range(num_params)] + ["cost"]
    df = pd.DataFrame(columns=columns)
    df.to_excel(excel_filename, index=False)

# Salva una nuova riga nel file Excel
def append_to_excel(params, cost, excel_filename):
    row = list(params) + [cost]
    new_df = pd.DataFrame([row])
    with pd.ExcelWriter(excel_filename, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
        new_df.to_excel(writer, index=False, header=False)


def create_bounds_constraints(num_params):
    constraints = []

    for i in range(num_params):
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i]})               # x[i] >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x, i=i: 2 * np.pi - x[i]})  # x[i] <= 2pi

    return constraints


# Funzione di costo modificata
def cost_func_estimator(params, ansatz, hamiltonian, estimator, excel_filename, opt_history):
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])
    results = job.result()[0]
    cost = results.data.evs

    opt_history.append((params.copy(), cost))

    append_to_excel(params, cost, excel_filename)  # salva su file
    print(f"Cost: {cost},   Params: {params}")
    return cost

# Funzione di ottimizzazione modificata
def params_optimization(init_params, cost_hamiltonian, backend, candidate_circuit, excel_filename):
    initialize_excel(len(init_params), excel_filename)  # inizializza il file con intestazioni
    opt_history = []  # <--- Qui si inizializza la lista

    constraints = create_bounds_constraints(len(init_params))

    with Session(backend=backend) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = 1000
        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = "XY4"
        estimator.options.twirling.enable_gates = True
        estimator.options.twirling.num_randomizations = "auto"

        result = minimize(
            cost_func_estimator,
            init_params,
            args=(candidate_circuit, cost_hamiltonian, estimator, excel_filename, opt_history),
            method="COBYLA",
            constraints=constraints,
            tol=1e-2,
        )

    return result, opt_history


### JOB ###

def read_last_params(excel_filename):
    try:
        df = pd.read_excel(excel_filename)
        last_row = df.iloc[-1]
        params = last_row[:-1].values
        cost = last_row[-1]
        return params, cost
    except FileNotFoundError:
        return None, None
    
def send_job(ansatz, hamiltonian, backend, new_params, job_tag="opt_job"):
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)
    pub = (ansatz, isa_hamiltonian, new_params)

    estimator = Estimator(backend=backend)
    job = estimator.run([pub], backend=backend)
    print(f"[INFO] Job sent: ID = {job.job_id()}")
    return job.job_id()

def fetch_and_record_result_job(job_id, new_params):
    provider = IBMProvider()
    job = provider.retrieve_job(job_id)

    while job.status() not in [JobStatus.DONE, JobStatus.ERROR, JobStatus.CANCELLED]:
        print(f"Waiting for job {job_id}... Status: {job.status().name}")
        time.sleep(5)

    if job.status() == JobStatus.DONE:
        result = job.result()
        cost = result[0].data.evs
        append_to_excel(new_params, cost)
        print(f"[RESULT] Cost: {cost} for params: {new_params}")
        return cost
    else:
        raise RuntimeError(f"Job {job_id} failed with status {job.status()}")
    

from scipy.optimize import minimize
def step_cobyla(current_params, cost_history):
    # Dummy objective to only compute one step
    def dummy_cost(params):
        return 0  # not evaluated

    result = minimize(
        dummy_cost,
        current_params,
        method="COBYLA",
        tol=1e-2,
        options={'maxiter': 1},
    )

    new_params = result.x
    return new_params

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def torino_native_ansatz(
    num_qubits: int,
    num_layers: int = 1,
    single_qubit_ops=("rz",),
    ent_gate: str = "cz",
    apply_hadamard: bool = True,
) -> QuantumCircuit:
    """
    Ansatz multi-strato per IBM Torino (processore Heron), con opzione per inizializzare
    la sovrapposizione uniforme |+>^n tramite H su ogni qubit.

    Gate nativi IBM Torino (Heron):
      - 1 qubit:  rz(θ), sx, x, id, delay
      - 2 qubit:  cz
      - operazioni: measure, reset
    (Opzionalmente disponibili fractional gates RX(θ), RZZ(θ).)

    Struttura del circuito:
      0) (Opzionale) Hadamard su tutti i qubit → sovrapposizione uniforme
      1) num_layers strati, ognuno composto da:
         - rotazioni 1q definite da single_qubit_ops
         - entanglement completo con ent_gate

    Parametri:
      - num_qubits: numero di qubit
      - num_layers: numero di strati
      - single_qubit_ops: tupla/lista di gate 1q tra {'rz', 'sx', 'x'}
      - ent_gate: gate 2q tra {'cz'}
      - apply_hadamard: se True, applica H a tutti i qubit all’inizio

    Ritorna:
      - QuantumCircuit con parametri θ[L, q] per ogni gate RZ parametrico.
    """
    native_1q = {"rz", "sx", "x"}
    native_2q = {"cz"}
    sq_ops = tuple(map(str.lower, single_qubit_ops))
    if not set(sq_ops).issubset(native_1q):
        raise ValueError(f"single_qubit_ops deve essere sottoinsieme di {native_1q}")
    ent_gate = ent_gate.lower()
    if ent_gate not in native_2q:
        raise ValueError(f"ent_gate deve essere uno di {native_2q}")

    qc = QuantumCircuit(num_qubits, name="torino_native_ansatz")

    # (Opzionale) sovrapposizione uniforme con Hadamard
    if apply_hadamard:
        qc.h(range(num_qubits))
        qc.barrier(label="uniform-superposition")

    # Parametri per RZ: uno per (layer, qubit)
    theta = [ParameterVector(f"θ[{L}]", num_qubits) for L in range(num_layers)]

    for L in range(num_layers):
        # --- blocco 1q
        for q in range(num_qubits):
            for op in sq_ops:
                if op == "rz":
                    qc.rz(theta[L][q], q)
                elif op == "sx":
                    qc.sx(q)
                elif op == "x":
                    qc.x(q)
        qc.barrier(label=f"layer-{L}-1q")

        # --- entanglement completo (tutte le coppie)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                if ent_gate == "cz":
                    qc.cz(i, j)
        qc.barrier(label=f"layer-{L}-ent")

    return qc


### VARIATIONAL

def optimize_variational_circuit(params, circuit, cost_hamiltonian, shots, service, backend_noisy):


    with Session(backend=backend_noisy) as session:
        estimator = Estimator(mode=session)
        estimator.options.default_shots = shots

        # Set simple error suppression/mitigation options
        estimator.options.dynamical_decoupling.enable = True
        estimator.options.dynamical_decoupling.sequence_type = "XY4"
        estimator.options.twirling.enable_gates = True
        estimator.options.twirling.num_randomizations = "auto"

        start_time = time.time()
        result = minimize(
            cost_func_estimator_sim,
            params,
            args=(circuit, cost_hamiltonian, estimator),
            method="COBYLA",
            tol=1e-2,
        )

    opt_time = time.time() - start_time
    print(result)
    return opt_params, min_cost, opt_time, result

def sample_variational_circuit(optimized_circuit, shots, backend):
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = shots

    # Set simple error suppression/mitigation options
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_gates = True
    sampler.options.twirling.num_randomizations = "auto"

    pub = (optimized_circuit,)

    job = sampler.run([pub], shots=int(shots))

    counts_bin = job.result()[0].data.meas.get_counts()
    final_distribution_bin = {key: val / shots for key, val in counts_bin.items()}
    return final_distribution_bin



