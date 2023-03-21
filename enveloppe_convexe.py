import sys
import numpy as np
import math
import time
import matplotlib.pyplot as plt
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class droite:
    def __init__( self, p1, p2):
        self.a =  p2[1] - p1[1]
        self.b = -p2[0] + p1[0]
        self.c =  p1[1] * p2[0] - p1[0] * p2[1]

    def meme_cote(self, q1, q2) -> bool:
        return ( self.a * q1[0] + self.b * q1[1] + self.c ) * ( self.a * q2[0] + self.b * q2[1] + self.c )  > 0

def calcul_enveloppe( nuage_de_points : np.ndarray ) -> np.ndarray :
    enveloppe = []
    lst_nuage = list(nuage_de_points[:])
    lst_nuage.sort(key=lambda coord : coord[1])
    bas = lst_nuage.pop(0)
    enveloppe.append(bas)

    lst_nuage.sort(key=lambda coord : math.atan2(coord[1]-bas[1], coord[0]-bas[0]))

    lst_nuage.append(bas)

    while len(lst_nuage) > 0:
        enveloppe.append(lst_nuage.pop(0))

        while len(enveloppe)>=4:
            if not droite( enveloppe[-3], enveloppe[-2] ).meme_cote( enveloppe[-4], enveloppe[-1] ):
                enveloppe.pop(-2)
            else:
                break
    return np.array(enveloppe)

taille_nuage : int = 55440
nbre_repet   : int =     3
resolution_x : int = 1_000
resolution_y : int = 1_000




if len(sys.argv) > 1:
    taille_nuage = int(sys.argv[1])
if len(sys.argv) > 2:
    nbre_repet   = int(sys.argv[2])

enveloppe = None
nuage     = None


if rank == 0:
    t1 = time.time()

for r in range(nbre_repet):
    if rank == 0:
        nuage = np.array(np.array([[resolution_x * i * math.cos(48371.*i)/taille_nuage for i in range(taille_nuage)], [resolution_y * math.sin(50033./(i+1.)) for i in range(taille_nuage)]], dtype=np.float64).T)


        scatter_data = np.array_split(nuage, size)
    else:
        scatter_data = None

    sous_nuage = comm.scatter(scatter_data, root=0)

    local_enveloppe = calcul_enveloppe(sous_nuage)

    if (size>=8):
        if rank in [0,1,2,3]:
            comm.send(local_enveloppe, dest=rank+4)
            received_enveloppe = comm.recv(source=rank+4)
            local_enveloppe = calcul_enveloppe(np.concatenate((local_enveloppe, received_enveloppe)))
        elif rank in [4,5,6,7]:
            received_enveloppe = comm.recv(source=rank-4)
            comm.send(local_enveloppe, dest=rank-4)
            local_enveloppe = calcul_enveloppe(np.concatenate((local_enveloppe, received_enveloppe)))
    elif (size>=4):
        if rank in [0,1,4,5]:
            comm.send(local_enveloppe, dest=rank+2)
            received_enveloppe = comm.recv(source=rank+2)
            local_enveloppe = calcul_enveloppe(np.concatenate((local_enveloppe, received_enveloppe)))
        elif rank in [2,3,6,7]:
            received_enveloppe = comm.recv(source=rank-2)
            comm.send(local_enveloppe, dest=rank-2)
            local_enveloppe = calcul_enveloppe(np.concatenate((local_enveloppe, received_enveloppe)))
    else :     
        for i in range(size // 2):
            if rank == i:
                comm.send(local_enveloppe, dest=i+1)
                received_enveloppe = comm.recv(source=i+1)
                local_enveloppe = calcul_enveloppe(np.concatenate((local_enveloppe, received_enveloppe)))
            elif rank == i+1:
                received_enveloppe = comm.recv(source=i)
                comm.send(local_enveloppe, dest=i)
                local_enveloppe = calcul_enveloppe(np.concatenate((local_enveloppe, received_enveloppe)))

    final_enveloppe = comm.gather(local_enveloppe, root=0)

    if rank == 0:
        enveloppe = np.concatenate(final_enveloppe)
        enveloppe = calcul_enveloppe(enveloppe)


        file = open("test.txt", "w+")
        content = str(np.sort(enveloppe))
        file.write(content)
        file.close()



if rank == 0:
    t2 = time.time()
    print(f"Temps total : {(t2-t1)/nbre_repet}")

    plt.scatter(nuage[:,0], nuage[:,1])
    for i in range(len(enveloppe[:])-1):
        plt.plot([enveloppe[i,0],enveloppe[i+1,0]], [enveloppe[i,1], enveloppe[i+1,1]], 'bo', linestyle="-")
    plt.show()




    if (taille_nuage == 55440):
        ref = np.loadtxt("enveloppe_convexe_55440.ref")
        try:
            np.testing.assert_allclose(ref, enveloppe)
            print("Verification pour 55440 points: OK")
        except AssertionError as e:
            print(e)
            print("Verification pour 55440 points: FAILED")
