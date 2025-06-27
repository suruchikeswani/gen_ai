# filename: divisible_by_7.py

# List to store numbers divisible by 7
divisible_by_7 = []

# Iterate through numbers from 1 to 50
for number in range(1, 51):
    if number % 7 == 0:
        divisible_by_7.append(number)

# Print the numbers divisible by 7
print("Numbers between 1 and 50 that are divisible by 7:", divisible_by_7)