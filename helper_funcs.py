import numpy as np
from mod_class import Mod
from random import randint


def add_noise(encoded_message: np.ndarray, mod: int, prob_noise: int) -> np.ndarray:
    counter = 0
    for i in range(len(encoded_message)):
        for j in range(len(encoded_message[i])):
            if randint(1, 100) < prob_noise:
                counter += 1
                encoded_message[i, j] = Mod(randint(0, mod-1), mod)
    return encoded_message, counter


def str_to_bits(s):
    return np.array([Mod(int(b), 2) for char in s for b in f"{ord(char):08b}"])

def bits_to_str(bits):
    bits = [b.a for b in bits]  # Convert Mod objects to ints
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]
    return ''.join(chr(int(''.join(map(str, c)), 2)) for c in chars if len(c) == 8)
