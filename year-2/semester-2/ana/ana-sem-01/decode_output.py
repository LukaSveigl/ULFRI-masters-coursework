
from ana_sem_01_2less import get_var, generate_triplets

if __name__ == '__main__':

    import sys
    # Read the k and n variables from the command line arguments
    if len(sys.argv) != 3:
        print("Usage: python decode_output.py <k> <n>")
        sys.exit(1)

    k = int(sys.argv[1])
    n = int(sys.argv[2])

    if k < 1 or n < 1:
        print("Invalid size. Must be a positive integer.")
        sys.exit(1)

    # Read the file output.txt
    with open("output.txt", "r") as file:
        lines = file.readlines()

    # Find line that starts with "s" and check if it contains "SATISFIABLE" or "UNSATISFIABLE"
    for line in lines:
        if line.startswith("s"):
            if "UNSATISFIABLE" in line:
                print("The problem is UNSATISFIABLE.")
            elif "SATISFIABLE" in line:
                print("The problem is SATISFIABLE.")
            else:
                print("Unknown status.")
            break
    else:
        print("No status line found.")

    # Find all lines that start with v and extract not-negated variables
    variables = []
    for line in lines:
        if line.startswith("v"):
            # Split the line into parts and filter out negative variables
            parts = line.split()
            for part in parts[1:]:
                if not part.startswith('-'):
                    variables.append(part)


    triplets = generate_triplets(n)
    triplet_vars = {triplet: [] for triplet in triplets}

    for triplet_index in range(len(triplets)):
        for sequence_index in range(k):
            triplet_vars[triplets[triplet_index]].append(get_var(triplet_index, sequence_index, k))

    # Reverse the encoding of the variables into a triplet
    # The variables are encoded as follows: triplet_index * k + sequence_index + 1
    # We extract the triplets from the variables - the triplet_vars contains a mapping of 
    # triplet: [variables]
    # For each triplet, we check if any of its variables are in the variables list
    decoded_triplets = []
    for triplet, vars in triplet_vars.items():
        if any(str(var) in variables for var in vars):
            decoded_triplets.append(triplet)

    # Print the decoded triplets
    print("Decoded triplets:")
    for triplet in decoded_triplets:
        print(triplet)


    