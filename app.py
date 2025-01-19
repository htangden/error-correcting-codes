from mod_class import Mod
import numpy as np
from random import randint
from code_class import Code
from helper_funcs import ndarr_to_str, str_to_ndarr, add_noise



G = np.array([
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
], dtype=int)

mod = 2
code = Code(G, mod, verbose=False)


''' CAN ONLY HANDLE ENGLISH ALPHABET AND SPACES (7 bit chars ASCII) '''
input_message = input("Input: ").replace(" ", "|")


encoded_message = code.encode_message(str_to_ndarr(input_message, mod))
noisy_encoded_message = add_noise(encoded_message, mod, 5)
str_noisy_encoded_message = ndarr_to_str(noisy_encoded_message).replace("|", " ")

decoded_message = code.decode_message(noisy_encoded_message)
str_decoded_message = ndarr_to_str(decoded_message).replace("|", " ")

print(f"After noise - encoded: {str_noisy_encoded_message}")
print(f"After noise - decoded: {str_decoded_message}")



