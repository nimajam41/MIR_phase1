import sys

s = "111010100"
x = int(s, 2)
r = x.to_bytes(len(s) // 8 + 1, sys.byteorder)
bytes = []
# f = open("inp", 'rb')
# f.write(r)
# f.close()
f = open("inp", 'wb')
byte = f.read(1)
while byte:
    bytes += [byte]
    byte = f.read(1)
f.close()
print(bytes)


