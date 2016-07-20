def extended_euclid(number_one, number_two):
    a = number_one
    b = number_two
    x2 = 1
    x1 = 0
    y2 = 0
    y1 = 1
    while b > 0:
        q = a // b
        r = a - q * b
        tmp1 = x2 - q * x1
        tmp2 = y2 - q * y1
        x2 = x1
        x1 = tmp1
        y2 = y1
        y1 = tmp2
        a = b
        b = r
    return a, x2, y2


# test
if extended_euclid(7, 11) != (1, -3, 2):
    print("error in extended_euclid with 7 11")
if extended_euclid(19, 11) != (1, -4, 7):
    print("error in extended_euclid with 19 11")