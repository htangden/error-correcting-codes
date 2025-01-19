import numpy as np
from mod_class import Mod
import json
from gauss import gauss_elimination_inverse_mod
import itertools

class Code:
    def __init__(self, generating_matrix: np.ndarray, prime: int, compute_coset_leader = True, verbose = False):

        self.prime = prime
        self.G = np.array([[Mod(int(value), self.prime) for value in row] for row in generating_matrix])
        (self.m, self.n) = np.shape(self.G)
        self.verbose = verbose

        if verbose:
            print(f"CODE:\n [{self.m}x{self.n}] mod {self.prime}\n")
            print("GENERATING MATRIX BEFORE NORMALIZING:")
            print(self.G, "\n")
        self.normalize_gen_matrix()

        if verbose:
            print("AFTER NORMALIZING:")
            print(self.G, "\n")

        self.create_control_matrix()

        if verbose:
            print("CREATED CONTROL MATRIX:")
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

    def __init__(self, r: int, compute_coset_leader = False, verbose = False):
        self.r = r
        super().__init__()
    
    def gen_hamming_gen_matrix(self):
        pass


