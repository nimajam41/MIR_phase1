import math, sys


def create_gamma_code(number, col):  # col is "title" or "description"
    gamma_code = ""
    if col == "title":
        gamma_code += "0"
    elif col == "description":
        gamma_code += "1"

    binary_of_number = bin(number)[2:]

    if number == 0:
        gamma_code += "0"
    else:
        right_section = binary_of_number[1:]
        for i in range(len(right_section)):
            gamma_code += "1"
        gamma_code += "0"
        gamma_code += right_section
    return int(gamma_code, 2).to_bytes(math.ceil(len(gamma_code) / 8), sys.byteorder)


def decode_gamma_code(number):
    string_number = str(format(int.from_bytes(number, sys.byteorder), 'b'))
    col_bit = string_number[0]
    col = None
    if col_bit == "0":
        col = "title"
    else:
        col = "description"
    gamma_code = string_number[1:]
    count_of_one = 0
    for i in range(len(gamma_code)):
        if gamma_code[i] == "1":
            count_of_one += 1
        else:
            break
    decoded_gamma_code_to_number = "1"
    decoded_gamma_code_to_number += gamma_code[count_of_one + 1:]
    return int(decoded_gamma_code_to_number, 2), col


print(create_gamma_code(1, "title"))
print(decode_gamma_code(b'\x00'))
