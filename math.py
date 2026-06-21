from mpmath import mp

mp.dps = 28  # Set decimal precision

def is_int(num):
    # Check if the number is real and if it's equal to its integer form
    if num.imag == 0:  # Ensure the number is real
        return num == int(num)
    return False  # If the number is complex, it's not an integer

num = 1000
other = 996
for i in range(62,63):
    result = mp.ceil(mp.sqrt(i*2+1))

# Ensure result*result is not smaller than num
    difference = result**2 -( i*2+1)

    if not is_int(mp.sqrt(difference))and difference !=0 :
        print(i*2+1)
        print(result)
        print(difference)
        print(is_int(mp.sqrt(difference)))
if is_int(mp.sqrt(mp.ceil(mp.sqrt(other))**2-other)):
    print(other)

