def calculate_statistics(numbers):
    if not numbers: 
        return None, None, None, None, None

    total_sum = 0
    cumulative_sum = []
    min_value = numbers[0]
    max_value = numbers[0]

    for i, num in enumerate(numbers):
        total_sum += num
        cumulative_sum.append(total_sum)
        if num < min_value:
            min_value = num
        if num > max_value:
            max_value = num

    average = total_sum / len(numbers)

    return total_sum, cumulative_sum, average, min_value, max_value

numbers = list(map(int, input("Enter numbers separated by spaces: ").split()))

print(numbers)

total_sum, cumulative_sum, average, min_value, max_value = calculate_statistics(numbers)

print(f"Sum: {total_sum}")
print(f"Cumulative Sum: {cumulative_sum}")
print(f"Average: {average}")
print(f"Min: {min_value}")
print(f"Max: {max_value}")