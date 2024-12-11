from code_class import Code
import numpy as np
from random import randint

def ndarr_to_str(arr: np.ndarray) -> str:
    return ''.join(np.array([chr(int(''.join(map(str, letter[:7])), 2)) for letter in arr]))

G = np.array([
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
], dtype=int)

code = Code(G, compute_coset_leader=True, verbose=True)


''' CAN ONLY HANDLE ENGLISH ALPHABET AND SPACES (7 bit chars ASCII) '''
input_message = input("Input: ").replace(" ", "|")

chunks = np.array([np.array([element for element in str(format(ord(char), 'b'))], dtype=int) for char in input_message], dtype=int)
encoded_message = code.encode_message(chunks)

print("oh no, noise!!!")

for i in range(len(encoded_message)):
    for j in range(len(encoded_message[i])):
        if randint(1, 100) < 5:
            encoded_message[i, j] = int(not bool(encoded_message[i, j]))


decoded_message = code.decode_message(encoded_message)
str_decoded_message = ndarr_to_str(decoded_message).replace("|", " ")
str_encoded_message = ndarr_to_str(encoded_message).replace("|", " ")
print(f"After noise - encoded: {str_encoded_message}")
print(f"After noise - decoded: {str_decoded_message}")
