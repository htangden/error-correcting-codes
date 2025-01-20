import numpy as np
from mod_class import Mod
import json
from gauss import gauss_elimination_inverse_mod
import itertools
import sys
from math import floor
from scipy.special import binom


class Code:
    def __init__(self, generating_matrix: np.ndarray, prime: int, compute_coset_leader = True, verbose = False):

        self.prime = prime
        if not isinstance(generating_matrix[0][0], Mod):
            self.G = np.array([[Mod(int(value), self.prime) for value in row] for row in generating_matrix])
        (self.m, self.n) = np.shape(self.G)
        self.verbose = verbose

        if verbose:
            print(f"CODE:\n [{self.m}x{self.n}] mod {self.prime}\n")
            print("GENERATING MATRIX BEFORE NORMALIZING:")
            print(self.G, "\n")

        arr = np.array([[val.a for val in row] for row in self.G[:,:self.m]])
        print("a", arr)
        arr_inv = np.linalg.inv(arr)
        print(arr_inv)
        self.normalize_gen_matrix()

        if verbose:
            print("AFTER NORMALIZING:")
            print(self.G, "\n")

        self.create_control_matrix()

        if verbose:
            print("CONTROL MATRIX:")
            print(self.H, "\n")

        self.coset_leaders = {}
        if compute_coset_leader:
            self.compute_coset_leaders()
            if verbose:
                print("COMPUTATION COSET LEADERS COMPLETE")
                print(self.coset_leaders, "\n")
        else:
            with open('data/coset_leader.json', 'r') as file:
                self.coset_leaders = json.load(file)

        self.seperation = self.compute_seperation()
        if verbose:
            print(f"SEPERATION:\n{self.seperation}\n")

        self.pack_density = self.compute_pack_density()
        if verbose:
            print(f"PACKING DENSITY\n{int(self.pack_density*100)}%")


    # make gen matrix of type [I | A]
    def normalize_gen_matrix(self):
        A = self.G[:,:self.m]
        A_inv = gauss_elimination_inverse_mod(A)
        self.G = A_inv @ self.G

    # from gen matrix create control matrix [-A | I]
    def create_control_matrix(self):
        if self.verbose:
            print(f"A:\n{self.G[:,self.m:]}\n")
            print(f"-At:\n{-self.G[:,self.m:].T}\n")

        neg_at = -self.G[:,self.m:].T
        identity = np.array([[Mod(1 if i == j else 0, self.prime) for j in range(self.n-self.m)] for i in range(self.n-self.m)], dtype=Mod)
        self.H = np.hstack((neg_at, identity))


    def compute_coset_leaders(self):
        points =  [[Mod(value, self.prime) for value in point] for point in list(itertools.product(range(self.prime), repeat=self.n))]

        for point in points:
            syndrome = self.compute_syndrome(point)
            if self.get_coset_leader(syndrome) is None:
                key = "".join([str(val.a) for val in syndrome])
                value = "".join([str(val.a) for val in point])
                self.coset_leaders[key] = value


            if self.weight(point) < self.weight(self.get_coset_leader(syndrome)):
                key = "".join([str(val.a) for val in syndrome])
                value = "".join([str(val.a) for val in point])
                self.coset_leaders[key] = value

        self.create_coset_file()

    def get_coset_leader(self, syndrome: np.ndarray) -> np.ndarray | None:
        key = "".join([str(val.a) for val in syndrome])
        try:
            coset_leader = list(self.coset_leaders[key])
            return np.array([Mod(int(value), self.prime) for value in coset_leader], dtype = Mod)
        except KeyError:
            return None
        
    def compute_seperation(self):
        points =  [[Mod(value, self.prime) for value in point] for point in list(itertools.product(range(self.prime), repeat=self.m))]
        all_weigths = [self.weight(point @ self.G) for point in points if self.weight(point @ self.G) != 0]
        return min(all_weigths)
    
    def compute_pack_density(self):
        amount_of_words = self.prime**self.m
        space = self.prime**self.n
        circle_sum = 0
        for i in range(floor((self.seperation-1)/2) + 1):
            circle_sum += binom(self.n, i)*(self.prime-1)**i
        return (amount_of_words*circle_sum)/space

    def weight(self, point: np.ndarray) -> int:
        return sum([val.a for val in point])

    def create_coset_file(self):
        with open("data/coset_leader.json", 'w') as file:
            json.dump(self.coset_leaders, file)

    def is_in_code(self, point) -> bool:
        return not np.any(point @ self.H.T)
    
    def compute_syndrome(self, point) -> np.ndarray:
        return point @ self.H.T
    
    def encode_message(self, message: np.ndarray) -> np.ndarray:
        return np.array([chunk @ self.G for chunk in message])
    
    def decode_message(self, message: np.ndarray) -> np.ndarray:
        return np.array([self.decode_chunk(chunk) for chunk in message], dtype=Mod)

    def decode_chunk(self, message: np.ndarray) -> np.ndarray:
        if self.is_in_code(message):
            return message[:self.m]
        else:
            syndrome = self.compute_syndrome(message)
            return message[:self.m] + self.get_coset_leader(syndrome)[:self.m]
    
class Hamming_Code(Code):

    def __init__(self, size: list[int, int], prime: int, compute_coset_leader = True, verbose = False):
        self.prime = prime
        self.m = size[0]
        self.n = size[1]
        if 2**(self.n-self.m)-1 != self.n:
            sys.exit("Incorrect [m, n] values fo hamming code. Must satisfy 2^(n-m) - 1 = n") 

        self.gen_hamming_gen_matrix()

        super().__init__(self.G, self.prime, compute_coset_leader=compute_coset_leader, verbose=verbose)


    
    def gen_hamming_gen_matrix(self):
        syndromes =  [[Mod(value, self.prime) for value in point] for point in list(itertools.product(range(self.prime), repeat=self.n-self.m))]
        a = -np.array([syndrome for syndrome in syndromes if self.weight(syndrome)>1])

        identity = np.array([[Mod(1 if i == j else 0, self.prime) for j in range(self.m)] for i in range(self.m)], dtype=Mod)
        self.G = np.hstack((identity, a))


class Reed_Muller_code(Code):

    def __init__(self, size: int, prime: int, compute_coset_leader = True, verbose = False):
        self.prime = prime
        start_gen_matrix = np.array([[Mod(0, self.prime), Mod(1, self.prime)], [Mod(1, self.prime), Mod(0, self.prime)]])
        self.G = self.create_gen_matrix(start_gen_matrix, 0, size-2)
        super().__init__(self.G, self.prime, compute_coset_leader, verbose)

    def create_gen_matrix(self, gen_matrix : np.ndarray, counter : int, stop : int):
        if counter == stop:
            return gen_matrix
        
        ones = np.array([Mod(1, self.prime) for _ in range(len(gen_matrix[0]))])
        zeros = np.array([Mod(0, self.prime) for _ in range(len(gen_matrix[0]))])

        new_gen_matrix = np.vstack(((np.hstack((gen_matrix, gen_matrix))), (np.hstack((zeros, ones)))))
        return self.create_gen_matrix(new_gen_matrix, counter+1, stop)

if __name__ == "__main__":
    rmC = Reed_Muller_code(3, 2, verbose=True)
