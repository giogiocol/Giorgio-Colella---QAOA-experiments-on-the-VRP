import pandas as pd
import numpy as np
import tabulate as tab
import random
import networkx as nx
import matplotlib.pyplot as plt
import csv
import math

import dimod
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
import neal

from qiskit_optimization import QuadraticProgram
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

### READ COORDINATES FROM CSV AND SAVE THE DISTANCE MATRIX ###

def read_coord_from_csv(filepath):
    coordinates = {}
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')  # Cambia il delimitatore se serve
        for row in reader:
            index = int(row[reader.fieldnames[0]])  # Prima colonna (es. '1', '2'…)
            x = float(row['x'])
            y = float(row['y'])
            coordinates[index] = (x, y)
    return coordinates


def compute_distance_matrix(coord_dict):
    keys = sorted(coord_dict.keys())
    n = len(keys)

    # Inizializza matrice n x n con zeri
    distance_matrix = {}

    for i in range(n):
        for j in range(n):
            x1, y1 = coord_dict[keys[i]]
            x2, y2 = coord_dict[keys[j]]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distance_matrix[(i,j)] = dist
    return distance_matrix

def plot_coordinates(coordinates):
    """
    Plot points from a dictionary of coordinates.

    Args:
        coordinates (dict): Dictionary where keys are indices and values are (x, y) tuples.
    """
    # Extract x and y values from the dictionary
    x_values = [coord[0] for coord in coordinates.values()]
    y_values = [coord[1] for coord in coordinates.values()]
    
    # Plot points
    plt.scatter(x_values, y_values, color='blue', label='Machines')
    plt.scatter(0, 0, color='red', label='I/O point', marker='x')
    
    # Annotate points with their indices
    for index, (x, y) in coordinates.items():
        plt.text(x, y, str(index), fontsize=9, ha='right')
    
    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Scatter Plot of Coordinates')
    plt.legend()
    #plt.grid(True)
    plt.show()

### VARIABLES DICTIONARY ###
def varaible_dictionaries(N):
    # Create a dictionary to map from (i, j) to the flattened index k
    X_ij_to_k = {}

    # Optional: create reverse mapping from k to (i, j) if needed
    X_k_to_ij = {}

    # Generate the mappings and variables
    p = 0
    k = 0  # Flat index counter
    for i in range(N):
        for j in range(N):
            # Compute flat index using the formula
            if i != j:
                if j<i:
                    t = j
                else:
                    t = j-1

                p = t

                k = (i)*(N-1) + p
            
            # Store the mapping
                X_ij_to_k[(i, j)] = k
                X_k_to_ij[k] = (i, j)

    return X_ij_to_k, X_k_to_ij

### CONSTRAINTS MATRIX ###
def build_matrix_A(N, X_ij_to_k):

    #CONSTRAINT 0
    A_C0 = np.empty((0, N*(N-1)))

    row = np.zeros(N*(N-1))
    for j in range(N):
        if j != 0:
            row[X_ij_to_k[(0, j)]] = 1

    A_C0 = np.vstack((A_C0, row))

    row = np.zeros(N*(N-1))
    for i in range(N):
        if i != 0:
            row[X_ij_to_k[(i, 0)]] = 1

    A_C0 = np.vstack((A_C0, row))

    #CONSTRAINT 1
    A_C1 = np.empty((0, N*(N-1)))

    for i in range(1, N):
        row = np.zeros(N*(N-1))

        for j in range(N):
            if i != j:
                row[X_ij_to_k[(i, j)]] = 1

        A_C1 = np.vstack((A_C1, row))

    #CONSTRAINT 2
    A_C2 = np.empty((0, N*(N-1)))

    for j in range(1, N):
        row = np.zeros(N*(N-1))

        for i in range(N):
            if i != j:
                row[X_ij_to_k[(i, j)]] = 1

        A_C2 = np.vstack((A_C2, row))

    #CONSTRAINT 3
    A_C3 = np.empty((0, N*(N-1)))
    for i in range(N):
        for j in range(N):
            if i != j:
                row = np.zeros(N*(N-1))
                row[X_ij_to_k[(i, j)]] = 1
                row[X_ij_to_k[(j, i)]] = 1
                A_C3 = np.vstack((A_C3, row))



    #Matrix composition
    A = np.vstack((A_C0, A_C1, A_C2, A_C3))

    return A, A_C0, A_C1, A_C2, A_C3


def build_vect_C(N, X_ij_to_k, d_ij): #penalty is the coefficient for the penalty component of the obj. function

    C = np.zeros((N*(N-1), N*(N-1)))

    for i in range(N):
        for j in range(N):
            if i != j:
                C[X_ij_to_k[(i, j)], X_ij_to_k[i, j]] = d_ij[(i, j)]

    return C

def build_matrix_P(A0, A1, A2, A3, d_ij, alpha=1): 

    p = alpha*max(d_ij.values())
    P0 = p*np.ones(A0.shape[0])
    P1 = p*np.ones(A1.shape[0])
    P2 = p*np.ones(A2.shape[0])
    P3 = p*np.ones(A3.shape[0])
    P = np.concatenate((P0, P1, P2, P3), axis=0)

    P = np.diag(P)

    return P

def build_vect_b(A, q):
    b = np.ones(A.shape[0]) 
    b[0] = q
    b[1] = q

    return b

# ADJUSTED FUNCTIONS

def build_matrix_A_adj(N, X_ij_to_k):

    #CONSTRAINT 0
    A_C0 = np.empty((0, N*(N-1)))

    row = np.zeros(N*(N-1))
    for j in range(N):
        if j != 0:
            row[X_ij_to_k[(0, j)]] = 1

    A_C0 = np.vstack((A_C0, row))

    row = np.zeros(N*(N-1))
    for i in range(N):
        if i != 0:
            row[X_ij_to_k[(i, 0)]] = 1

    A_C0 = np.vstack((A_C0, row))

    #CONSTRAINT 1
    A_C1 = np.empty((0, N*(N-1)))

    for i in range(1, N):
        row = np.zeros(N*(N-1))

        for j in range(N):
            if i != j:
                row[X_ij_to_k[(i, j)]] = 1

        A_C1 = np.vstack((A_C1, row))

    #CONSTRAINT 2
    A_C2 = np.empty((0, N*(N-1)))

    for j in range(1, N):
        row = np.zeros(N*(N-1))

        for i in range(N):
            if i != j:
                row[X_ij_to_k[(i, j)]] = 1

        A_C2 = np.vstack((A_C2, row))

    #Matrix composition
    A = np.vstack((A_C0, A_C1, A_C2))

    return A, A_C0, A_C1, A_C2


def build_vect_C_adj(N, X_ij_to_k, d_ij, pen): #penalty is the coefficient for the penalty component of the obj. function

    C = np.zeros((N*(N-1), N*(N-1)))

    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                C[X_ij_to_k[(i, j)], X_ij_to_k[i, j]] = d_ij[(i, j)]

                C[X_ij_to_k[(i, j)], X_ij_to_k[j, i]] = pen*max(d_ij.values()) 


    return C


def build_matrix_P_adj(A0, A1, A2, d_ij, alpha): 

    p = alpha*max(d_ij.values())
    P0 = p*np.ones(A0.shape[0])
    P1 = p*np.ones(A1.shape[0])
    P2 = p*np.ones(A2.shape[0])
    P = np.concatenate((P0, P1, P2), axis=0)

    P = np.diag(P)

    return P

### QUBO FORMULATION ###

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


def compute_QUBO_terms(C: np.ndarray,
                            A: np.ndarray,
                            P: np.ndarray,
                            b: np.ndarray):
    """
    Calcola i termini Q, L_t e c per l’espressione
        y = x^T Q x + L^T x + c

    Input:
    - C: matrice (n×n)
    - A: matrice (m×n)
    - P: matrice (m×m)
    - b: vettore colonna (m,)

    Output:
    - Q: matrice quadratica (n×n)
    - L: vettore riga dei coefficienti lineari (1×n), pari a L^T
    - c: scalare, termine costante
    """
    # termine quadratico
    Q = C + A.T @ P @ A

    # termine lineare trasposto (L^T)
    L_t = -2 * (b.T @ P @ A)    # shape: (1, n)

    # termine costante
    c = float(b.T @ P @ b)      # converti a scalare

    return Q, L_t, c

def build_qubo(Q: np.ndarray, L: np.ndarray) -> np.ndarray:
    """
    Build the symmetric QUBO matrix Q' = 0.5*(Q + Q^T) + diag(L)
    Inputs:
      Q : (n×n) real array (possibly non‑symmetric)
      L : (n,) or (n×1) array of linear bias coefficients
    Returns:
      Qp: (n×n) symmetric QUBO matrix for x^T Qp x
    """
    # 1) Symmetrize Q
    Q_sym = 0.5 * (Q + Q.T)
    # 2) Absorb the linear terms: diag(Q_sym) += L
    Qp = Q_sym.copy()
    np.fill_diagonal(Qp, np.diag(Qp) + L.flatten())  # in-place diag update :contentReference[oaicite:0]{index=0}
    return Qp

def build_QUBO_dict(Q_qubo, X_k_to_ijr):
    QUBO_dict = {}
    for i in range(Q_qubo.shape[0]):
        for j in range(Q_qubo.shape[1]):
            QUBO_dict[(X_k_to_ijr[i], X_k_to_ijr[j])] = Q_qubo[i, j]

    return QUBO_dict

def solve_qubo(Q_qubo, solver_type="quantum_annealing", **sampler_kwargs):
    """
    Solve a QUBO using various D-Wave Ocean samplers.

    Args:
        Q_qubo (dict or {(i,j): coefficient} or 2D-array):
            Your QUBO in any form accepted by dimod.BinaryQuadraticModel.from_qubo.
        solver_type (str): one of
            - "quantum_annealing": embed + send to D-Wave QPU
            - "simulated_annealing": run neal.SimulatedAnnealingSampler
            - "exact": dimod.ExactSolver (only for very small QUBOs)
            - "hybrid": D-Wave’s LeapHybridSampler
        **sampler_kwargs: extra keyword args passed to sampler.sample(),
            e.g. num_reads, chain_strength, label, time_limit, etc.

    Returns:
        dimod.SampleSet: the result sampleset from the chosen sampler.
    """
    # 1) Build a BQM from your QUBO
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q_qubo)

    # 2) Choose the sampler
    solver_type = solver_type.lower()
    if solver_type == "quantum_annealing":
        # EmbeddingComposite handles minor‐embedding for you
        sampler = EmbeddingComposite(DWaveSampler())
    elif solver_type == "simulated_annealing":
        sampler = neal.SimulatedAnnealingSampler()
    elif solver_type == "exact":
        sampler = dimod.ExactSolver()
    elif solver_type == "hybrid":
        sampler = LeapHybridSampler()
    else:
        raise ValueError(f"Unknown solver_type '{solver_type}'. "
                         "Choose from 'quantum_annealing', 'simulated_annealing', "
                         "'exact' or 'hybrid'.")

    # 3) Sample & return
    sampleset = sampler.sample(bqm, **sampler_kwargs)
    
    return sampleset


def from_problem_exname_to_cost_hamiltonian(excel_name, n_vehicles=2, pen=1):
    q = n_vehicles # Number of vehicles

    coord_dict = read_coord_from_csv(excel_name)

    N = len(coord_dict)

    dist_mat = compute_distance_matrix(coord_dict)

    dic_ij_to_k, dic_k_ij = varaible_dictionaries(N)

    A, A_C0, A_C1, A_C2 = build_matrix_A_adj(N, dic_ij_to_k)

    C = build_vect_C_adj(N, dic_ij_to_k, dist_mat, pen = pen)

    P = build_matrix_P_adj(A_C0, A_C1, A_C2, dist_mat, alpha=pen)

    b = build_vect_b(A, q)

    Q, L_t, c = compute_QUBO_terms(C, A, P, b)

    Q_qubo = build_qubo(Q, L_t)

    qubo_dict = build_QUBO_dict(Q_qubo, dic_k_ij)

    qp = from_qubo_to_qp(Q_qubo)

    cost_hamiltonian, offset = qp.to_ising()

    return cost_hamiltonian, N


### SOLUTION SAVE & PLOT ###

def extract_best_n(sol_sample, n, sorted_by='num_occurrences', reverse=True):
    """
    Estrae i migliori n campioni da un SampleSet ordinandoli secondo un criterio.

    Parametri:
    - sol_sample: oggetto SampleSet (es. da un'ottimizzazione D-Wave)
    - n: numero di campioni da estrarre
    - sorted_by: criterio di ordinamento ('energy', 'num_occurrences', 'sample')
    - reverse: se True ordina in modo decrescente

    Ritorna:
    - sol_ord: nuovo SampleSet contenente i n migliori campioni
    """
    # Estrai e ordina i dati
    data = list(sol_sample.data(['sample', 'energy', 'num_occurrences'], sorted_by=sorted_by, reverse=reverse))

    # Troncamento ai primi n
    data = data[:n]

    # Spezza i dati in liste
    samples = [d[0] for d in data]
    energies = [d[1] for d in data]
    occurrences = [d[2] for d in data]

    # Ricostruisci SampleSet ordinato
    sol_ord = dimod.SampleSet.from_samples(
        samples_like=samples,
        vartype=sol_sample.vartype,
        energy=energies,
        num_occurrences=occurrences
    )

    return sol_ord


def save_solution_matrix_to_csv(best, filename="soluzione_best_matrice_A+B.csv"):
    """
    Salva una matrice rappresentata da un dizionario 'best' in un file CSV.
    
    Parametri:
    - best: dict con chiavi (i, j) e valori 0 o 1
    - filename: nome del file CSV di output (default: "soluzione_best_matrice_A+B.csv")
    """
    if not best:
        raise ValueError("Il dizionario 'best' è vuoto.")

    # Determina la dimensione della matrice
    N = max(max(i, j) for i, j in best.keys()) + 1

    # Crea matrice NxN inizializzata a 0
    matrice = np.zeros((N, N), dtype=int)

    # Riempi la matrice con i valori del dizionario
    for (i, j), val in best.items():
        matrice[i][j] = val

    # Crea DataFrame con intestazioni e indici
    df = pd.DataFrame(matrice, columns=[f'{j}' for j in range(N)])
    df.index = [f'{i}' for i in range(N)]

    # Salva su CSV
    df.to_csv(filename, index=True)

import networkx as nx
import matplotlib.pyplot as plt

def plot_vrp_solution(best, coord_dict, title="Solution Graph Plot"):
    G = nx.DiGraph()  # Grafo orientato

    # Aggiungi nodi al grafo
    for node, (x, y) in coord_dict.items():
        G.add_node(node, pos=(x, y))
    
    # Aggiungi archi dalla soluzione
    for (i, j), value in best.items():
        if value == 1:
            G.add_edge(i, j)

    pos = nx.get_node_attributes(G, 'pos')

    plt.figure(figsize=(12, 10))

    # Colori dei nodi (nodo 0 verde chiaro, altri azzurri)
    node_colors = ['lightgreen' if node == 0 else 'lightblue' for node in G.nodes()]

    # Disegna i nodi (più grandi)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=800,
        node_color=node_colors,
        edgecolors='black',
        linewidths=1.5
    )

    # Disegna le etichette dei nodi (più grandi)
    nx.draw_networkx_labels(
        G, pos,
        font_size=14,
        font_weight='bold'
    )

    # Disegna gli archi (più spessi e con frecce più grandi)
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=25,
        width=2.5
    )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.grid(False)
    plt.show()


def compute_obj_function_value(sol_i, dist_mat, N):
    f_value = 0
    for i in range(N):
        for j in range(N):
            if i != j:
                if sol_i[(i, j)] == 1:
                    f_value += dist_mat[(i, j)]
    return f_value

def plot_solution_probabilities(sol_agg, num_reads, col_by, dist_mat, N):
    """
    Plot a histogram of the probabilities of obtaining each unique solution,
    sorted in descending order of probability and colored by energy.
    
    Parameters:
        sol_agg (dimod.SampleSet): Aggregated SampleSet (from .aggregate()).
        num_reads (int): Total number of reads (samples).
        col_by (str): energy or fo
    """
    # Estrai dati
    if col_by not in ['energy', 'fo']:
        raise ValueError("col_by must be either 'energy' or 'fo'.")
    
    if col_by == 'energy':
        color = sol_agg.record.energy

    elif col_by == 'fo':
        color = np.array([compute_obj_function_value(sol, dist_mat, N) for sol in sol_agg.samples()])

    occurrences = sol_agg.record.num_occurrences
    total_occurrences = num_reads
    
    # Calcola probabilità
    probabilities = occurrences / total_occurrences

    # Ordina per probabilità decrescente
    sorted_indices = np.argsort(-probabilities)
    sorted_probs = probabilities[sorted_indices]
    sorted_color = color[sorted_indices]

    # Crea mappa colori basata sull'energia
    norm = plt.Normalize(vmin=min(sorted_color), vmax=max(sorted_color))
    colors = plt.cm.viridis(norm(sorted_color))

    # Plot istogramma
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(sorted_probs)), sorted_probs, color=colors, edgecolor='k')

    # Colore con barra laterale
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])  # richiesto per ScalarMappable
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Energy')

    ax.set_xlabel('Solution Rank (by descending probability)')
    ax.set_ylabel('Probability')
    ax.set_title('Solution Probability Histogram Colored by Energy')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def check_validity(sol_i, N, q):
    valid = 1
    sum = 0

    for i in range(1,N):
        sum = 0
        for j in range(N):
            if i != j:
                sum += sol_i[(i,j)]
        if sum != 1:
            valid = 0

    for j in range(1,N):
        sum = 0
        for i in range(N):
            if i != j:
                sum += sol_i[(i,j)]
        if sum != 1:
            valid = 0

    sum = 0
    for j in range(1,N):
        sum += sol_i[(0,j)]

    if sum != q:
        valid = 0

    sum = 0
    for i in range(1,N):
        sum += sol_i[(i,0)]

    if sum != q:
        valid = 0

    for i in range(1, N):
        for j in range(1, N):
            if i != j:
                if (sol_i[(i,j)] + sol_i[(j,i)] > 1):
                    valid = 0

    return valid

def valid_percentage(sol_sample, N, q):
    count = 0
    for i in range(len(sol_sample)):
        sol_i = sol_sample.samples()[i]
        if check_validity(sol_i, N, q):
            count += 1
    return count / len(sol_sample) * 100

import matplotlib.pyplot as plt

def plot_frequency_hist(final_distribution_int, code_opt, code_feas):
    """
    Plot frequency histogram from a dict {code: frequency}.
    
    Bars:
      - Optimal solution (code_opt): green
      - Feasible solutions (code_feas): blue
      - All others: light gray
    
    Title and labels are in English.
    """
    # Ensure feasible codes is a set for fast lookup
    feas_set = set(code_feas) if code_feas is not None else set()
    
    # Sort items by frequency descending, then by code for stability
    items = sorted(final_distribution_int.items(), key=lambda kv: (-kv[1], str(kv[0])))
    codes = [str(k) for k, _ in items]
    freqs = [v for _, v in items]
    
    # Prepare bar colors
    colors = []
    for k_str, (k, _) in zip(codes, items):
        if k == code_opt:
            colors.append("#2ecc71")  # green for optimal
        elif k in feas_set:
            colors.append("#3498db")  # blue for feasible
        else:
            colors.append("#d3d3d3")  # light gray for others
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(codes, freqs, color=colors, edgecolor="black", linewidth=0.5)
    plt.title("Measurement Outcome Probability Distribution")
    plt.xlabel("Measurement Outcome (reversed bitstring)")
    plt.ylabel("Probability")
    plt.xticks(rotation=45, ha="right")
    plt.legend(["Optimal Solutions", "Feasible Solutions", "Infeasible Solutions"], loc="upper right")
    plt.tight_layout()
    plt.show()