import numpy as np
import json
import itertools
import sys
from math import floor
from random import randint
from scipy.special import binom

class Mod:
    def __init__(self, a: int, n: int):
        self.a = a%n
        self.n = n

    def __add__(self, other_number):
        if isinstance(other_number, Mod):
            if self.n == other_number.n:
                return Mod((self.a + other_number.a) % self.n, self.n)
        else:
            return False
    
    def __sub__(self, other):
        if isinstance(other, Mod):
            return Mod(self.a - other.a, self.n)
        return False
        
    def __mul__(self, other):
        if isinstance(other, Mod):
            if self.n == other.n:
                return Mod((self.a * other.a) % self.n, self.n)
        else: 
            return False

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Mod):
            inv = pow(other.a, -1, self.n)
            return Mod(self.a * inv, self.n)
        return False
    
    def __neg__(self):
        return Mod(-self.a, self.n)
        
    def __str__(self):
        return f"({self.a} mod {self.n})"
    
    def __eq__(self, other):
        if isinstance(other, Mod):
            return self.a == other.a and self.n == other.n
        return False

    def __repr__(self):
        return f"({self.a}m{self.n})"
    


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
                print(",\n".join(f"{key}: {value}" for key, value in self.coset_leaders.items()) + "\n")

        else:
            with open('data/coset_leader.json', 'r') as file:
                self.coset_leaders = json.load(file)

        self.seperation = self.compute_seperation()
        if verbose:
            print(f"SEPERATION:\n{self.seperation}\n")

        self.pack_density = self.compute_pack_density()
        if verbose:
            print(f"PACKING DENSITY\n{int(self.pack_density*100)}%\n")


    # make gen matrix of type [I | A]
    def normalize_gen_matrix(self):
        A = self.G[:,:self.m]
        A_inv = self.gauss_elimination_inverse_mod(A)
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
    
    def pad_bits(self, bits):
        pad_length = (self.m - (len(bits) % self.m))
        return np.concatenate([bits, np.array([Mod(0, self.prime)] * pad_length)])

    def weight(self, point: np.ndarray) -> int:
        return sum([val.a for val in point])

    def create_coset_file(self):
        with open("data/coset_leader.json", 'w') as file:
            json.dump(self.coset_leaders, file)

    def is_in_code(self, point) -> bool:
        return not np.any(point @ self.H.T)
    
    def compute_syndrome(self, point) -> np.ndarray:
        return point @ self.H.T
    
    def str_to_bits(self, s):
        return np.array([int(b) for char in s for b in f"{ord(char):08b}"])

    def bits_to_str(self, bits):
        chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
        return ''.join(chr(int(''.join(map(str, c)), 2)) for c in chars if len(c) == 8)

    def encode_str(self, message_str: str) -> np.ndarray:
        return self.encode_message(self.str_to_bits(message_str))

    def encode_message(self, message: np.ndarray) -> np.ndarray:
        message = [Mod(val, self.prime) for val in message]

        message = self.pad_bits(message).reshape(-1, self.m)
        encoded = []
        for chunk in message:
            encoded += list(chunk @ self.G)

        encoded = [val.a for val in encoded]
        return np.array(encoded)
    
    def decode_to_str(self, message: np.ndarray) -> str:
        return self.bits_to_str(self.decode_message(message))

    def decode_message(self, message: np.ndarray) -> np.ndarray:
        message = np.array([Mod(val, self.prime) for val in message])
        message = message.reshape(-1, self.n)
        decoded = np.array([self.decode_chunk(chunk) for chunk in message.reshape(-1, self.n)], dtype=Mod).flatten()
        return [int(val.a) for val in decoded]

    def decode_chunk(self, message: np.ndarray) -> np.ndarray:
        if self.is_in_code(message):
            return message[:self.m]
        else:
            syndrome = self.compute_syndrome(message)
            return message[:self.m] + self.get_coset_leader(syndrome)[:self.m]
    
    def add_noise(self, encoded_message: np.ndarray, prob_noise: int) -> np.ndarray:
        counter = 0

        for i in range(len(encoded_message)):
            if randint(1, 100) < prob_noise:
                counter += 1
                encoded_message[i] = randint(0, self.prime-1)
        
        encoded_str = self.bits_to_str([encoded_message[i] for i in range(len(encoded_message)) if i%self.n<self.m ])

        return encoded_message, encoded_str, counter
    
    def gauss_elimination_inverse_mod(self, matrix):
        n = matrix.shape[0]

        # Ensure the matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Only square matrices can be inverted.")

        # Create an augmented matrix with the identity matrix on the right-hand side
        prime = matrix[0, 0].n  # Assuming all elements are Mod with the same prime
        identity = np.array([[Mod(1 if i == j else 0, prime) for j in range(n)] for i in range(n)], dtype=object)
        augmented = np.hstack((matrix, identity))

        for i in range(n):
            # Pivot: Make sure the diagonal element is not zero
            if augmented[i, i] == Mod(0, prime):
                # Find a row below the current one with a nonzero value in this column and swap
                for j in range(i + 1, n):
                    if augmented[j, i] != Mod(0, prime):
                        augmented[[i, j]] = augmented[[j, i]]
                        break
                else:
                    raise ValueError("Matrix is singular and cannot be inverted.")

            # Normalize the pivot row
            augmented[i] = [x / augmented[i, i] for x in augmented[i]]

            # Eliminate the other entries in this column
            for j in range(n):
                if i != j:
                    factor = augmented[j, i]
                    augmented[j] = [augmented[j, k] - factor * augmented[i, k] for k in range(2 * n)]

        # The right-hand side of the augmented matrix is now the inverse
        inverse = augmented[:, n:]
        return inverse

class Hamming_Code(Code):

    def __init__(self, size: list[int, int], compute_coset_leader = True, verbose = False):
        self.m = size[0]
        self.n = size[1]
        if 2**(self.n-self.m)-1 != self.n:
            sys.exit("Incorrect [m, n] values fo hamming code. Must satisfy 2^(n-m) - 1 = n") 

        self.gen_hamming_gen_matrix()

        super().__init__(self.G, 2, compute_coset_leader=compute_coset_leader, verbose=verbose)

    
    def gen_hamming_gen_matrix(self):
        syndromes =  [[Mod(value, 2) for value in point] for point in list(itertools.product(range(2), repeat=self.n-self.m))]
        a = -np.array([syndrome for syndrome in syndromes if self.weight(syndrome)>1])

        identity = np.array([[Mod(1 if i == j else 0, 2) for j in range(self.m)] for i in range(self.m)], dtype=Mod)
        self.G = np.hstack((identity, a))


class Reed_Muller_code(Code):

    def __init__(self, size: int, compute_coset_leader = True, verbose = False):

        start_gen_matrix = np.array([[Mod(1 if i == j else 0, 2) for j in range(3)] for i in range(3)], dtype=Mod)
        start_gen_matrix = np.hstack((start_gen_matrix, np.array([[Mod(1, 2)], [Mod(1, 2)], [Mod(1, 2)]])))
        
        self.G = self.create_gen_matrix(start_gen_matrix, 0, size-2)
        super().__init__(self.G, 2, compute_coset_leader, verbose)

    def create_gen_matrix(self, gen_matrix : np.ndarray, counter : int, stop : int):
        if counter == stop:
            return gen_matrix
        
        ones = np.array([Mod(1, 2) for _ in range(len(gen_matrix[0]))])
        zeros = np.array([Mod(0, 2) for _ in range(len(gen_matrix[0]))])

        new_gen_matrix = np.vstack(((np.hstack((gen_matrix, gen_matrix))), (np.hstack((zeros, ones)))))
        return self.create_gen_matrix(new_gen_matrix, counter+1, stop)


if __name__ == "__main__":
    rmC = Reed_Muller_code(3, 2, verbose=True)
