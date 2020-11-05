import sys

s = "10111010100"
x = int(s, 2)
r = x.to_bytes(len(s) // 8 + 1, sys.byteorder)
bytes = []
# f = open("inp", 'wb')
# f.write(r)
# f.close()
f = open("inp", 'rb')
byte = f.read(1)
while byte:
    bytes += [byte]
    byte = f.read(1)
f.close()
bytes.reverse()
s2 = ""
for byte in bytes:
    a = int.from_bytes(byte, sys.byteorder)
    s2 += format(a, 'b')
print(s2)
