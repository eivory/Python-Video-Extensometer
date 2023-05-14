import csv

# Function to convert pixels to inches using the pixels per inch value
def pixels_to_inches(pixels, pixels_per_inch): # pixels and pixels_per_inch are parameters
    return pixels / pixels_per_inch 

# Read the CSV file
filename = input("Enter the CSV file name: ")
with open(filename, 'r') as file:
    reader = csv.reader(file) 
    data = list(reader)

# Get the header row
header = data[0]

# Get the data rows
rows = data[1:]

# Get the initial length from the first value in the length column
initial_length = float(rows[0][1])

# Get the user input length
user_length = float(input("Enter the user length in inches: "))

# Calculate the pixels per inch
pixels_per_inch = initial_length / user_length

# Convert the lengths in pixels to inches
lengths_inches = [] # Create an empty list to store the lengths in inches
for row in rows: # Iterate through the rows
    length_pixels = float(row[1]) # Get the length in pixels
    length_inches = pixels_to_inches(length_pixels, pixels_per_inch) # Convert the length in pixels to inches
    lengths_inches.append(length_inches) # Add the length in inches to the list

# Create the output CSV file name
output_filename = filename.replace('.csv', '_in.csv')

# Write the results to the output CSV file
with open(output_filename, 'w', newline='') as file: # Open the output CSV file
    writer = csv.writer(file) # Create a CSV writer
    writer.writerow(header) # Write the header row
    for i in range(len(rows)): # Iterate through the rows
        writer.writerow([rows[i][0], lengths_inches[i]]) # Write the row with the length in inches

print(f"Output saved to {output_filename}") # Print the output file name
