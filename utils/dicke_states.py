# dicke_states.py — Qiskit 2.0 ready (tutte le correzioni)

import numpy as np
import matplotlib.pyplot as plt
import scipy  # opzionale

# Qiskit 2.x
from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RYGate
from qiskit.visualization import plot_histogram

# Aer 2.x
from qiskit_aer import AerSimulator


##################################################################
# util
##################################################################

def show_figure(fig):
    """
    Mostra una figura Matplotlib anche se non è l'ultima cella.
    """
    new_fig = plt.figure()
    new_mngr = new_fig.canvas.manager
    new_mngr.canvas.figure = fig
    fig.set_canvas(new_mngr.canvas)
    plt.show()


##################################################################
# simulator helpers
##################################################################

def test_circuit_qasm(qc: QuantumCircuit, figsize=(12, 4)):
    """Simula il circuito con misure usando AerSimulator (QASM).
    Il circuito in input non deve avere misure: vengono aggiunte qui.
    Ritorna il plot degli esiti (counts).
    """
    qc_meas = qc.copy()
    qc_meas.measure_all()

    backend = AerSimulator()
    backend.set_options(seed_simulator=42)

    tqc = transpile(qc_meas, backend)
    job = backend.run(tqc, shots=100_000)
    results = job.result()
    counts = results.get_counts()

    print(f"\nNumber of elements in result superposition: {len(counts)}\n")
    return plot_histogram(counts, title="Results", figsize=figsize)


def test_circuit_sv(qc: QuantumCircuit, print_stuff=False, figsize=(12, 4)):
    """Simula lo statevector ideale del circuito (nessuna misura).
    Ritorna un istogramma delle probabilità per i bitstring.
    """
    sv = Statevector.from_instruction(qc)
    probs_dict = sv.probabilities_dict()

    if print_stuff:
        amps = sv.data
        probs = np.abs(amps) ** 2
        print(f"Statevector:\n{amps}\n")
        print(f"Probabilities:\n{probs}\n")

    print(f"\nNumber of elements in result superposition: {len(probs_dict)}\n")
    return plot_histogram(probs_dict, title="Results", figsize=figsize)


##################################################################
# gates elementari compositi (come Instruction, non Gate)
##################################################################

def gate_i(n: int, draw: bool = False):
    """Gate a 2 qubit (i): CX - cRY(theta) - CX, theta = 2 * arccos(sqrt(1/n))."""
    qc_i = QuantumCircuit(2)

    qc_i.cx(0, 1)

    theta = 2 * np.arccos(np.sqrt(1 / n))
    cry = RYGate(theta).control(ctrl_state="1")
    qc_i.append(cry, [1, 0])

    qc_i.cx(0, 1)

    g = qc_i.to_instruction()   # <- importante per evitare unknown instruction
    g.name = "i"                # nome ASCII semplice per Aer
    g.label = "$(i)$"           # solo per disegno

    if draw:
        show_figure(qc_i.draw("mpl"))
    return g


def gate_ii_l(l: int, n: int, draw: bool = False):
    """Gate a 3 qubit (ii)_l: CX - ccRY(theta) - CX, theta = 2 * arccos(sqrt(l/n))."""
    qc_ii = QuantumCircuit(3)

    qc_ii.cx(0, 2)

    theta = 2 * np.arccos(np.sqrt(l / n))
    ccry = RYGate(theta).control(num_ctrl_qubits=2, ctrl_state="11")
    qc_ii.append(ccry, [2, 1, 0])

    qc_ii.cx(0, 2)

    g = qc_ii.to_instruction()
    g.name = f"ii_{l}"
    g.label = f"$(ii)_{l}$"

    if draw:
        show_figure(qc_ii.draw("mpl"))
    return g


def gate_scs_nk(n: int, k: int, draw: bool = False):
    """Gate SCS_{n,k} (Def. 3): un (i) e (k-1) gate (ii)_l per l=2..k, su (k+1) qubit."""
    qc_scs = QuantumCircuit(k + 1)

    qc_scs.append(gate_i(n), [k - 1, k])

    for l in range(2, k + 1):
        qc_scs.append(gate_ii_l(l, n), [k - l, k - l + 1, k])

    g = qc_scs.to_instruction()
    g.name = f"SCS_{n}_{k}"
    g.label = "SCS$_{" + f"{n},{k}" + "}$"

    if draw:
        show_figure(qc_scs.decompose().draw("mpl"))
    return g


##################################################################
# blocchi (come Instruction, non Gate)
##################################################################

def first_block(n: int, k: int, l: int, draw: bool = False):
    """Primo blocco del prodotto di unitarie (Lemma 2)."""
    qr = QuantumRegister(n)
    qc_first_block = QuantumCircuit(qr)

    n_first = l - k - 1
    n_last = n - l

    idxs_scs = list(range(n))

    if n_first != 0:
        idxs_scs = idxs_scs[n_first:]
        # identità sui primi n_first qubit
        for q in qr[:n_first]:
            qc_first_block.id(q)

    if n_last != 0:
        idxs_scs = idxs_scs[:-n_last]
        qc_first_block.append(gate_scs_nk(l, k), idxs_scs)
        # identità sugli ultimi n_last qubit
        for q in qr[-n_last:]:
            qc_first_block.id(q)
    else:
        qc_first_block.append(gate_scs_nk(l, k), idxs_scs)

    # label per disegno; nome semplice per Aer
    str_operator = "$1^{\\otimes " + f'{n_first}' + "} \\otimes$ SCS$_{" + f"{l},{k}" + "} \\otimes 1^{\\otimes " + f"{n_last}" + "}$"

    g = qc_first_block.to_instruction()
    g.name = f"FIRST_{l}"
    g.label = str_operator

    if draw:
        show_figure(qc_first_block.decompose().draw("mpl"))
    return g


def second_block(n: int, k: int, l: int, draw: bool = False):
    """Secondo blocco del prodotto di unitarie (Lemma 2)."""
    qr = QuantumRegister(n)
    qc_second_block = QuantumCircuit(qr)

    n_last = n - l
    idxs_scs = list(range(n))

    if n_last != 0:
        idxs_scs = idxs_scs[:-n_last]
        qc_second_block.append(gate_scs_nk(l, l - 1), idxs_scs)
        for q in qr[-n_last:]:
            qc_second_block.id(q)
    else:
        qc_second_block.append(gate_scs_nk(l, l - 1), idxs_scs)

    str_operator = "SCS$_{" + f"{l},{l-1}" + "} \\otimes 1^{\\otimes " + f"{n_last}" + "}$"

    g = qc_second_block.to_instruction()
    g.name = f"SECOND_{l}"
    g.label = str_operator

    if draw:
        show_figure(qc_second_block.decompose().draw("mpl"))
    return g


##################################################################
# builder principale
##################################################################

def dicke_state(n: int, k: int, draw: bool = False, barrier: bool = False, only_decomposed: bool = False) -> QuantumCircuit:
    """Costruisce il circuito che prepara lo stato di Dicke |D^n_k>.

    Args:
        n: numero di qubit.
        k: peso di Hamming (numero di 1).
        draw: se True, disegna il circuito (a vari livelli di decomposizione).
        barrier: se True, inserisce barrier tra i blocchi.
        only_decomposed: se True, disegna solo la decomposizione a 3 livelli.
    """
    qr = QuantumRegister(n)
    qc = QuantumCircuit(qr)

    # stato iniziale |1> sugli ultimi k qubit
    if k > 0:
        qc.x(qr[-k:])

    if barrier:
        qc.barrier()

    # Primo termine (l = k+1..n) in ordine inverso
    for l in range(k + 1, n + 1)[::-1]:
        qc.append(first_block(n, k, l), range(n))
        if barrier:
            qc.barrier()

    # Secondo termine (l = 2..k) in ordine inverso
    for l in range(2, k + 1)[::-1]:
        qc.append(second_block(n, k, l), range(n))
        if barrier:
            qc.barrier()

    # Disegni (opzionali)
    if draw:
        if only_decomposed:
            show_figure(qc.decompose().decompose().decompose().draw("mpl"))
        else:
            show_figure(qc.draw("mpl"))
            print()
            show_figure(qc.decompose().draw("mpl"))
            print()
            show_figure(qc.decompose().decompose().draw("mpl"))
            print()
            show_figure(qc.decompose().decompose().decompose().draw("mpl"))

    return qc


##################################################################
# transpile helper per Aer / Sampler
##################################################################

def transpile_for_aer(qc: QuantumCircuit, optimization_level: int = 1) -> QuantumCircuit:
    """Decompone ricorsivamente e transpila verso AerSimulator.
    Usare il risultato per Sampler(backend=AerSimulator) o backend.run.
    """
    # Decomposizione aggressiva per eliminare tutte le istruzioni composte
    decomp = qc
    for _ in range(5):
        decomp = decomp.decompose()
    backend = AerSimulator()
    tqc = transpile(decomp, backend, optimization_level=optimization_level)
    return tqc
