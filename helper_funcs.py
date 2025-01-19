import numpy as np
from mod_class import Mod
from random import randint

def ndarr_to_str(arr: np.ndarray) -> str:
    return ''.join([chr(int(''.join(map(str, [val.a for val in letter[:7]])), 2)) for letter in arr])

def str_to_ndarr(message: str, mod: int) -> np.ndarray:
    return np.array([np.array([Mod(int(element), mod) for element in str(format(ord(char), 'b'))], dtype=Mod) for char in message], dtype=Mod)

def add_noise(encoded_message: np.ndarray, mod: int, prob_noise: int) -> np.ndarray:
    for i in range(len(encoded_message)):
        for j in range(len(encoded_message[i])):
            if randint(1, 100) < prob_noise:
                new_num = randint(0, mod-1)
                encoded_message[i, j] = Mod(new_num, mod)
    return encoded_message