from code_class import Code, Hamming_Code

hCode = Hamming_Code([11, 15])

input_message = input("Input: ")

encoded_message = hCode.encode_str(input_message)

noisy_encoded_message, noisy_str, nbr_noise = hCode.add_noise(encoded_message, 4)

decoded_string = hCode.decode_to_str(noisy_encoded_message)


print(f"Original Message: {input_message}")
print(f"Before decoding: {noisy_str}")
print(f"Decoded Message: {decoded_string}")




