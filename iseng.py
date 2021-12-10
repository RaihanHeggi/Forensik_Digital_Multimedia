with open("input.in", "r") as f:
    contents = f.readlines()

contents = [s.rstrip() for s in contents]
counter_data = 0
iterasi = 1
for x in contents[1:]:
    if counter_data == 0:
        A = int(x)
        counter_data += 1
    elif counter_data == 1:
        B = int(x)
        counter_data += 1
    else:
        K = int(x)

        divisible_k = 0
        for i in range(A, B):
            if i % K == 0:
                print(i)
                divisible_k += 1

        # print("Case ", iterasi, ":", divisible_k)
        break
        iterasi += 1
        counter_data = 0
