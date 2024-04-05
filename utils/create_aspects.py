#create_aspects.py
#creates: create_aspects.txt, an aspects table for use in data_functions

def calculate_compatible_sizes(start, end, step):
    square_dims = [i for i in range(start, end + 1, step)]
    compatible_sizes = {}
    for i, square_dim in enumerate(square_dims):
        compatible_sizes[square_dim] = []
        min_pixels = 0 if i == 0 else square_dims[i-1]**2
        max_pixels = square_dim**2
        for width in range(64, square_dim * 4, 64):
            for height in range(64, square_dim * 4, 64):
                if min_pixels < width * height <= max_pixels:
                    aspect_ratio = width / height
                    if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                        if not width * height <= (start - step) ** 2:
                            compatible_sizes[square_dim].append([width, height])
        # Removing duplicates by converting the list of lists to a set of tuples and back to a list of lists
        compatible_sizes[square_dim] = [list(tup) for tup in set(tuple(item) for item in compatible_sizes[square_dim])]
        # Sorting the list of lists by the total pixels
        compatible_sizes[square_dim].sort(key=lambda x: x[0]*x[1], reverse=True)
    return compatible_sizes

file_path = "create_aspects.txt"

#Define compatible sizes
start_dim = 256
end_dim = 2048
step_dim = 64
max_aspect_ratio = 4.0 / 1
min_aspect_ratio = 1 / max_aspect_ratio

#run
compatible_sizes = calculate_compatible_sizes(start_dim, end_dim, step_dim)

#write the results to a text file
with open(file_path, "w") as file:
    for square_dim, sizes in compatible_sizes.items():
        file.write(f"{square_dim}: {sizes},\n")

#printout
print(f"  --{file_path} created")