import numpy as np
from code_class import Code, Hamming_Code
from helper_funcs import add_noise, str_to_bits, bits_to_str


hCode = Hamming_Code([11, 15], verbose=True)


input_message = input("Input: ")
bit_message = str_to_bits(input_message)

encoded_message = hCode.encode_message(bit_message)

noisy_encoded_message, nbr_noise = add_noise(encoded_message, 2, 4)
noisy_str = bits_to_str(np.array([letter[:11] for letter in noisy_encoded_message]).flatten())
print(f"Changed {nbr_noise} bits.")

decoded_message = hCode.decode_message(noisy_encoded_message)
decoded_string = bits_to_str(decoded_message)



print(f"Original Message: {input_message}")
print(f"Before decoding: {noisy_str}")
print(f"Decoded Message: {decoded_string}")




