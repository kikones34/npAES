import numpy as np

""" Printing utils """
from contextlib import contextmanager
@contextmanager
def np_line_width(width):
    oldoptions = np.get_printoptions()
    np.set_printoptions(linewidth=width)
    yield
    np.set_printoptions(**oldoptions)

vhex = np.vectorize("0x{:02x}".format)

""" Data """

sbox = np.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
], dtype=np.uint8)

sbox_inv = np.array([
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
], dtype=np.uint8)

mix_matrix = np.array([
    [2, 3, 1, 1],
    [1, 2, 3, 1],
    [1, 1, 2, 3],
    [3, 1, 1, 2],
], dtype=np.uint8)

mix_matrix_inv = np.array([
    [14, 11, 13,  9],
    [ 9, 14, 11, 13],
    [13,  9, 14, 11],
    [11, 13,  9, 14],
], dtype=np.uint8)

rcon_vals = np.array([
    # 0th index is unused, value doesn't matter
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
], dtype=np.uint8, ndmin=2).T
rcon = np.pad(rcon_vals, ((0, 0), (0, 3)))


""" Helper functions """

def gen_params(key):
    Nk = len(key) // 4 # number of words in the cipher key
    Nr = Nk + 6 # number of rounds
    return Nk, Nr

# multiplies two polynomials in GF(2)[X] / (x^8 + x^4 + x^3 + x + 1)    
# copypasted, TODO: understand why it works http://writes.co.de/2014/01/21/bitwise_galois/
def pmul(p1, p2):
    res = 0
    for i in range(8):
        if p2 & 1:
            res ^= p1
        hi_bit_set = p1 & 0x80
        p1 <<= 1
        p1 &= 0xFF
        if hi_bit_set:
            p1 ^= 0x1b
        p2 >>= 1
    return res
vpmul = np.vectorize(pmul)

# multiplies a vector by a matrix using pmul
pmul_vec = lambda v, m: np.apply_along_axis(lambda row: np.bitwise_xor.reduce(vpmul(row, v)), 1, m)


""" Key scheduler functions """

RotWord = lambda w: np.roll(w, -1)
SubWord = np.vectorize(lambda w: sbox[w])

def generate_keys(cipher_key, Nk, Nr):
    nwords = (Nr + 1) * 4
    words = np.zeros((nwords, 4), dtype=np.uint8)
    words[0:Nk] = np.array(cipher_key).reshape(Nk, -1)
    for i in range(Nk, nwords):
        if Nk == 8 and i % Nk == 4:
            words[i] = words[i-Nk] ^ SubWord(words[i-1])
        elif i % Nk == 0:
            words[i] = words[i-Nk] ^ SubWord(RotWord(words[i-1])) ^ rcon[i//Nk]
        else:
            words[i] = words[i-Nk] ^ words[i-1]
    keys = np.transpose(words.reshape(Nr + 1, 4, 4), (0, 2, 1))
    return keys


""" AES encrypt / decrypt functions """

SubBytes    = np.vectorize(lambda b: sbox[b])
SubBytesInv = np.vectorize(lambda b: sbox_inv[b])
shift_rows_dual = lambda m, sgn: np.array([np.roll(row, sgn*i) for i, row in enumerate(m)], dtype=np.uint8)
ShiftRows    = lambda m: shift_rows_dual(m, -1)
ShiftRowsInv = lambda m: shift_rows_dual(m,  1)
mix_columns_dual = lambda m, mix_m: np.apply_along_axis(lambda v: pmul_vec(v, mix_m), 0, m)
MixColumns    = lambda m: mix_columns_dual(m, mix_matrix)
MixColumnsInv = lambda m: mix_columns_dual(m, mix_matrix_inv)

# AES encode
def encode(input_bytes, cipher_key):
    state_matrix = np.array(input_bytes, dtype=np.uint8).reshape(4, 4).T
    Nk, Nr = gen_params(cipher_key)
    keys = generate_keys(cipher_key, Nk, Nr)

    # add first round key
    state_matrix ^= keys[0]

    # rounds
    for i in range(1, Nr+1):
        state_matrix = SubBytes(state_matrix)
        state_matrix = ShiftRows(state_matrix)
        if i < Nr:
            state_matrix = MixColumns(state_matrix)
        state_matrix ^= keys[i]

    return state_matrix.flatten(order='F')

# AES decode
def decode(input_bytes, cipher_key):
    state_matrix = np.array(input_bytes, dtype=np.uint8).reshape(4, 4).T
    Nk, Nr = gen_params(cipher_key)
    keys = generate_keys(cipher_key, Nk, Nr)

    # add last round key
    state_matrix ^= keys[Nr]

    # rounds
    for i in reversed(range(0, Nr)):
        state_matrix = ShiftRowsInv(state_matrix)
        state_matrix = SubBytesInv(state_matrix)
        state_matrix ^= keys[i]
        if i > 0:
            state_matrix = MixColumnsInv(state_matrix)
        
    return state_matrix.flatten(order='F')


""" Tests """

def do_test(input_bytes, cipher_key, expected_res, func):
	res = func(input_bytes, cipher_key)
	with np_line_width(np.inf):
		print(f"Input:    {vhex(input_bytes)}")
		print(f"Key:      {vhex(cipher_key)}")
		print(f"Output:   {vhex(res)}")
		print(f"Expected: {vhex(expected_res)}")
	if np.all(res == expected_res):
		print("PASS")
	else:
		print("FAIL")


print("128 bit encode")
input_bytes =  [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]
cipher_key =   [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f]
expected_res = [0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a]
do_test(input_bytes, cipher_key, expected_res, encode)
print()

print("128 bit decode")
input_bytes =  [0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a]
cipher_key =   [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f]
expected_res = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]
do_test(input_bytes, cipher_key, expected_res, decode)
print()


print("192 bit encode")
input_bytes =  [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]
cipher_key =   [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
                0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
expected_res = [0xdd, 0xa9, 0x7c, 0xa4, 0x86, 0x4c, 0xdf, 0xe0, 0x6e, 0xaf, 0x70, 0xa0, 0xec, 0x0d, 0x71, 0x91]
do_test(input_bytes, cipher_key, expected_res, encode)
print()

print("192 bit decode")
input_bytes =  [0xdd, 0xa9, 0x7c, 0xa4, 0x86, 0x4c, 0xdf, 0xe0, 0x6e, 0xaf, 0x70, 0xa0, 0xec, 0x0d, 0x71, 0x91]
cipher_key =   [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
                0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]
expected_res = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]
do_test(input_bytes, cipher_key, expected_res, decode)
print()


print("256 bit encode")
input_bytes =  [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]
cipher_key =   [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
                0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f]
expected_res = [0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf, 0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49, 0x60, 0x89]
do_test(input_bytes, cipher_key, expected_res, encode)
print()

print("256 bit decode")
input_bytes =  [0x8e, 0xa2, 0xb7, 0xca, 0x51, 0x67, 0x45, 0xbf, 0xea, 0xfc, 0x49, 0x90, 0x4b, 0x49, 0x60, 0x89]
cipher_key =   [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
                0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f]
expected_res = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]
do_test(input_bytes, cipher_key, expected_res, decode)
print()
