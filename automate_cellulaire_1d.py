import numpy as np
import time
import matplotlib.pyplot as plt
import sys
from mpi4py import MPI

nombre_cas   : int = 256
nb_cellules  : int = 360  # Cellules fantomes
nb_iterations: int = 360

compute_time = 0.
display_time = 0.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def save_as_md(cells, symbols='⬜⬛'):
    res = np.empty(shape=cells.shape, dtype='<U')
    res[cells==0] = symbols[0]
    res[cells==1] = symbols[1]
    np.savetxt(f'resultat_{num_config:03d}.md', res, fmt='%s', delimiter='', header=f'Config #{num_config}', encoding='utf-8')

def save_as_png(cells):
    fig = plt.figure(figsize=(nb_iterations/10., nb_cellules/10.))
    ax = plt.axes()
    ax.set_axis_off()
    ax.imshow(cells[:, 1:-1], interpolation='none', cmap='RdPu')
    plt.savefig(f"resultat_{num_config:03d}.png", dpi=100, bbox_inches='tight')
    plt.close()

configs = range(nombre_cas)

if rank == 0:
    t0 = time.time()

for num_config in configs[rank::size]: #On itère sur les configs d'indice égal à rank modulo size
    cells = np.zeros((nb_iterations, nb_cellules+2), dtype=np.int16)
    cells[0, (nb_cellules+2)//2] = 1
    for iter in range(1, nb_iterations):
        vals = np.left_shift(1, 4*cells[iter-1, 0:-2]
                             + 2*cells[iter-1, 1:-1]
                             + cells[iter-1, 2:])
        cells[iter, 1:-1] = np.logical_and(np.bitwise_and(vals, num_config), 1)

    save_as_md(cells)
#    save_as_png(cells)




if rank == 0:
    exec_time = time.time() - t0
    print(f"Execution time: {exec_time} seconds")
