import csv

input_file = 'first_passage_time_log.csv'
output_file = 'first_passage_time_stripped.csv'

# Process the CSV file
with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    for row in reader:
        # Check if the last element has brackets and remove them
        if row[-1].startswith('[') and row[-1].endswith(']'):
            row[-1] = row[-1][1:-1]  # Strip the brackets
        writer.writerow(row)

print(f"Processed file saved as {output_file}")