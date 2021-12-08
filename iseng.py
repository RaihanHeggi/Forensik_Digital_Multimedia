import math


def g(stringValue):
    i = 0
    new_string = ""
    while i < (len(stringValue) - 1):
        new_string += stringValue[i + 1]
        i += 1
    return new_string


def f(stringValue):
    if len(stringValue) == 0:
        return ""
    elif len(stringValue) == 1:
        return stringValue
    else:
        return f(g(stringValue)) + stringValue[0]


def h(integerValue, stringValue):
    while integerValue != 1:
        if integerValue % 2 == 0:
            integerValue = integerValue / 2
        else:
            integerValue = 3 * integerValue + 1
        stringValue = f(stringValue)
    return stringValue


def pow_val(x, y):
    if y == 0:
        return 1
    else:
        return x * pow(x, y - 1)


print(h(1, "fruits"))
print(h(2, "fruits"))
print(h(5, "fruits"))
print(h(pow_val(2, 1000000000000000), "fruits"))
print(h(pow_val(2, 9831050005000007), "fruits"))
