from code_class import Hamming_Code


hCode = Hamming_Code([4, 7])

input_string = input("Input: ")

encoded_message = hCode.encode_str(input_string)

noisy_encoded_message, noisy_string, nbr_noise = hCode.add_noise(encoded_message, 5)
print(f"Changed {nbr_noise} bits.")

decoded_string = hCode.decode_to_str(noisy_encoded_message)

print(f"Original Message: {input_string}")
print(f"Before decoding: {noisy_string}")
print(f"Decoded Message: {decoded_string}")





