# import sys

# def main(lines):
#     # このコードは標準入力と標準出力を用いたサンプルコードです。
#     # このコードは好きなように編集・削除してもらって構いません。
#     # ---
#     # This is a sample code to use stdin and stdout.
#     # Edit and remove this code as you like.

#     for i, v in enumerate(lines):
#         print("line[{0}]: {1}".format(i, v))

# if __name__ == '__main__':
#     lines = []
#     for l in sys.stdin:
#         lines.append(l.rstrip('\r\n'))
#     main(lines)

import sys
# from itertools import combinations

def combination(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # nCk = nC(n-k)を利用
    c = 1
    for i in range(k):
        c = c * (n - i) // (i + 1)
    return c

def main():

    # print("line[{0}]: {1}".format(i, v))
    # n, m = map(int, v.split())
    n = 2
    m = 2
    tiles = n * m
    l = [n,m]
    # print(i)
    print(l)
    result = combination(tiles, tiles // 2)
    for x in result:
        print(x)
    # print(len(result))
    # print(tiles)

if __name__ == '__main__':
    main()