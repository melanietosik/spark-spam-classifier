import csv

ham = []
spam = []

with open("data/spam.csv", "r") as csvfile:

    # Skip header
    next(csvfile)

    # Read CSV
    read = csv.reader(csvfile)
    for row in read:
        label, text = row[:2]
        if label == "ham":
            ham.append(text)
        elif label == "spam":
            spam.append(text)
        else:
            input("This shouldn't happen...")

    # Write output files
    with open("data/ham.txt", "w") as hamfile:
        for sms in ham:
            hamfile.write(sms + "\n")

    with open("data/spam.txt", "w") as spamfile:
        for sms in spam:
            spamfile.write(sms + "\n")
