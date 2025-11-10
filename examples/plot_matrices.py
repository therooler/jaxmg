import matplotlib as mpl
import matplotlib.pyplot as plt
from jaxmg._cyclic_1d import plot_block_to_cyclic

if __name__ == "__main__":
    with mpl.rc_context(
        {"font.size": 15, "font.family": "STIXGeneral", 'mathtext.fontset': 'stix'}
    ):
        N = 100  # - 2**12
        NRHS = 1
        T_A = 4
        fig, axs = plot_block_to_cyclic(N=N, T_A=T_A, ndev=4, N_rows=6)
        fig.savefig("../resources/mat.png", dpi=300)

        N = 12  # - 2**12
        NRHS = 1
        T_A = 2
        fig, axs = plot_block_to_cyclic(N=N, T_A=T_A, ndev=2, N_rows=6)
        fig.savefig("../resources/mat_example.png", dpi=300)
