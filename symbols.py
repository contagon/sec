NUM_EACH = 1000
FIRST_CHAR = 65

A = lambda k: 0 * NUM_EACH + k
B = lambda k: 1 * NUM_EACH + k
C = lambda k: 2 * NUM_EACH + k
D = lambda k: 3 * NUM_EACH + k
E = lambda k: 4 * NUM_EACH + k
F = lambda k: 5 * NUM_EACH + k
G = lambda k: 6 * NUM_EACH + k
H = lambda k: 7 * NUM_EACH + k
I = lambda k: 8 * NUM_EACH + k
J = lambda k: 9 * NUM_EACH + k
K = lambda k: 10 * NUM_EACH + k
L = lambda k: 11 * NUM_EACH + k
M = lambda k: 12 * NUM_EACH + k
N = lambda k: 13 * NUM_EACH + k
O = lambda k: 14 * NUM_EACH + k
P = lambda k: 15 * NUM_EACH + k
Q = lambda k: 16 * NUM_EACH + k
R = lambda k: 17 * NUM_EACH + k
S = lambda k: 18 * NUM_EACH + k
T = lambda k: 19 * NUM_EACH + k
U = lambda k: 20 * NUM_EACH + k
V = lambda k: 21 * NUM_EACH + k
W = lambda k: 22 * NUM_EACH + k
X = lambda k: 23 * NUM_EACH + k
Y = lambda k: 24 * NUM_EACH + k
Z = lambda k: 25 * NUM_EACH + k


def sym2str(s: int):
    i = s // NUM_EACH
    print(i)
    k = s % NUM_EACH
    return chr(FIRST_CHAR + i) + str(k)
