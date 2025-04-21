# Error correcting codes

Project meant for encoding messages such that they can be correctly decoded even after experiencing noise.  
The theoretical basis for the project is described in [Finite Fields and Error-Correcting Codes by Karl-Gustav Andersson](http://www.matematik.lu.se/matematiklu/personal/sigma/Andersson.pdf). 

### The Mod Class

All operations done on bits will be done in a finite field defined as operations modulo a prime number. Bits in this project are of type `Mod(a, n)` where `a` is the value and `n` the finite field prime number.

## The Code class

In order to encode or decode messages you need an object of type `Code`. 

<pre>
  class Code:
    def __init__(
                self, 
                generating_matrix: np.ndarray, 
                prime: int, 
                compute_coset_leader = True, 
                verbose = False
                ):

</pre>

- `prime`: refers to the finite field in which all calculations are to be done.  
- `compute_coset_leader`: set to `False` if there exists a file "data/coset_leader.json" which has correct coset leader information. Such a file is created when `compute_coset_leader=True`.  

Sample code:

<pre>
  G = np.array([
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
], dtype=int)

  code = Code(G, 2, verbose=True)
</pre>

## Encoding messages

Encoding is done with `Code`'s `encode_message()` method. 

<pre>
  class Code:
    def encode_message(self, message: np.ndarray) -> np.ndarray:
</pre>

The message is an list of bits, where each bit is of type Mod. To convert a string to the desired input format for `encode_message()` use the function `str_to_bits()` found in `helper_funcs.py`.

## Decoding message

Encoding is done with `Code`'s `decode_message()` method.

<pre>
  class Code:
    def decode_message(self, message: np.ndarray) -> np.ndarray:
</pre>

Where message is a list of encoded bits, type Mod. So long as the number of errors in a chunk is less than or equal to (σ-1)/2, where σ is the seperation of the code, it will decode the chunk correctly. To turn the output of `decode_message()` into a string use the function `bits_to_str()` found in `helper_funcs.py`.

### Other attributes of a Code
- `code.pack_density`: can be seen as how relatively good a code is. Perfect codes (for example hamming codes) have `pack_density=1`.
- `code.seperation`: the seperation of the code. Seperation is 3 for all hamming codes. 



# Sample code

<pre>
  import numpy as np
  from code_class import Code, Hamming_Code
  from helper_funcs import add_noise, str_to_bits, bits_to_str
  
  
  G = np.array([
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
], dtype=int)

  code = Code(G, 2)
  
  
  input_message = input("Input: ")
  bit_message = str_to_bits(input_message)
  
  encoded_message = code.encode_message(bit_message)
  
  noisy_encoded_message, nbr_noise = add_noise(encoded_message, 2, 4)
  noisy_str = bits_to_str(np.array([letter[:7] for letter in noisy_encoded_message]).flatten()) # the 7 comes from nbr rows in G
  print(f"Changed {nbr_noise} bits.")
  
  decoded_message = code.decode_message(noisy_encoded_message)
  decoded_string = bits_to_str(decoded_message)
  
  
  
  print(f"Original Message: {input_message}")
  print(f"Before decoding: {noisy_str}")
  print(f"Decoded Message: {decoded_string}")
</pre>

## Hamming codes

To create perfect binary codes of seperation 3 one can use objects of type `Hamming_Code`. 

<pre>
  class Hamming_Code(Code):
    def __init__(
                self, 
                size: list[int, int], 
                compute_coset_leader = True, 
                verbose = False):
</pre>

- `size`: the type of hamming code you wish to use. For a valid size = [m, n], the following must be true: 2^(n-m) - 1 = n.
-  Valid sizes can be calculated by plugging in integer values of r in [2^r − 1, 2^r − r − 1]

### Sample code
<pre>
hCode = Hamming_Code([11, 15])


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

</pre>
