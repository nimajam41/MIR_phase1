import pickle
from bitarray import bitarray
import sys
import bitstring
import math

gamma_positional_index = {"english": dict(), "persian": dict()}
main_positional_index = {"english": dict(), "persian": dict()}
decompress = {"english": dict(), "persian": dict()}


def create_variable_byte(number, col):  # col is "title" or "description"
    number = bin(number).replace("0b", "")

    while len(number) % 6 != 0:
        number = "0" + number
    result = ""
    byte_size = len(number) // 6
    for i in range(byte_size):
        if i == byte_size - 1:
            result += "1"
        else:
            result += "0"
        result += number[6 * i:6 * (i + 1)]
        if col == "title":
            result += "0"
        elif col == "description":
            result += "1"
    return int(result, 2).to_bytes(byte_size, sys.byteorder)  # returns bytes of data


def decode_variable_byte(number):
    number = format(int.from_bytes(number, sys.byteorder), 'b')
    while len(number) % 8 != 0:
        result = "0" + number
    byte_size = len(number) // 8
    result = ""
    for i in range(byte_size):
        result += number[8 * i + 1:8 * i + 7]
    col = (number[-1] == "0") * "title" + (number[-1] == "1") * "description"
    return int(result, 2), col


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


def positional_index_to_gamma_code(positional_index, gamma_positional_index):
    for term in positional_index.keys():
        for doc_id in positional_index[term].keys():
            if term not in gamma_positional_index.keys():
                gamma_positional_index[term] = dict()
            if doc_id not in gamma_positional_index[term].keys():
                gamma_positional_index[term][doc_id] = dict()
            if doc_id == "cf":
                gamma_positional_index[term]["cf"] = positional_index[term]["cf"]
                continue
            for col in positional_index[term][doc_id].keys():
                for i in range(len(positional_index[term][doc_id][col])):
                    if i == 0:
                        gamma_positional_index[term][doc_id] = [
                            create_gamma_code(positional_index[term][doc_id][col][i], col)]
                    else:
                        gamma_positional_index[term][doc_id] += [
                            create_gamma_code(positional_index[term][doc_id][col][i]
                                              - positional_index[term][doc_id][col][i - 1], col)]


def gamma_code_to_positional_index(gamma_positional_index, positional_index):
    dict(positional_index).clear()
    for term in gamma_positional_index.keys():
        for doc_id in gamma_positional_index[term].keys():
            if term not in positional_index.keys():
                positional_index[term] = dict()
            if doc_id not in positional_index[term].keys():
                positional_index[term][doc_id] = dict()
            if doc_id == "cf":
                positional_index[term]["cf"] = gamma_positional_index[term]["cf"]
                continue
            for i in range(len(gamma_positional_index[term][doc_id])):
                gap, col = decode_gamma_code(gamma_positional_index[term][doc_id][i])
                if col not in positional_index[term][doc_id].keys():
                    positional_index[term][doc_id][col] = [gap]
                else:
                    last_value = positional_index[term][doc_id][col][-1]
                    positional_index[term][doc_id][col] += [last_value + gap]


with open('positional_english_indexing', 'rb') as pickle_file:
    main_positional_index["english"] = pickle.load(pickle_file)
    pickle_file.close()

print(create_gamma_code(45, "title"))
print(decode_gamma_code(b'\xcd\x07'))