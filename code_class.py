import numpy as np
import json

class Code:
    # g - generating matrix
    # h - control matrix

    def __init__(self, generating_matrix: np.ndarray, compute_coset_leader = False, verbose=False):
        self.G = self.z_2_matrix(generating_matrix)
        (self.m, self.n) = np.shape(self.G)

        self.normalize_gen_matrix()
        self.create_control_matrix()

        self.coset_leaders = {}
        if compute_coset_leader:
            self.compute_coset_leaders()
            if verbose:
                print("COMPUTATION COSET LEADERS COMPLETE")
        else:
            with open('coset_leader.json', 'r') as file:
                self.coset_leaders = json.load(file)

    def normalize_gen_matrix(self):
        A = self.G[:,:self.m]
        A_inv = self.z_2_matrix(np.linalg.inv(A))
        self.G = (self.z_2_matrix(A_inv @ self.G))

    def create_control_matrix(self):
        neg_at = self.G[:,self.m:].T
        self.H = np.hstack((neg_at, np.eye(self.n-self.m)))

    def compute_coset_leaders(self):
        for point in np.array(np.meshgrid(*[[0, 1]] * self.n)).T.reshape(-1, self.n):
            syndrome = self.compute_syndrome(point)
            try:
                if sum(point) < sum(self.get_coset_leader(syndrome)):
                    self.coset_leaders[str(syndrome)] = ''.join(map(str, point))
            except KeyError:
                self.coset_leaders[str(syndrome)] = ''.join(map(str, point))
        self.create_coset_file()

    def get_coset_leader(self, syndrome: np.ndarray) -> np.ndarray:
        return np.array(list(self.coset_leaders[str(syndrome)]), dtype=int)

    def create_coset_file(self):
        with open("coset_leader.json", 'w') as file:
            json.dump(self.coset_leaders, file)


    def is_in_code(self, point) -> bool:
        return not np.any(self.z_2_point(point @ self.H.T))
    
    def compute_syndrome(self, point) -> np.ndarray:
        return self.z_2_point(point @ self.H.T)
    
    def encode_message(self, message: np.ndarray) -> np.ndarray:
        return np.array([self.z_2_point(chunk @ self.G) for chunk in message])
    
    def decode_message(self, message: np.ndarray) -> np.ndarray:
        return np.array([self.decode_chunk(chunk) for chunk in message], dtype=int)

    def decode_chunk(self, message: np.ndarray) -> np.ndarray:
        if self.is_in_code(message):
            return message[:self.m]
        else:
            syndrome = self.compute_syndrome(message)
            return self.z_2_point(message + self.get_coset_leader(syndrome))[:self.m]

    def z_2_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return np.array([[round(element)%2 for element in row] for row in matrix])

    def z_2_point(self, point: np.ndarray) -> np.ndarray:
        return np.array([round(element)%2 for element in point])
    