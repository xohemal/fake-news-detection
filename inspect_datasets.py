import pandas as pd

print("🔹 TRAIN SET:")
train = pd.read_csv('train.tsv', sep='\t')
print(train.head(), '\n')

print("🔹 VALID SET:")
valid = pd.read_csv('valid.tsv', sep='\t')
print(valid.head(), '\n')

print("🔹 TEST SET:")
test = pd.read_csv('test.tsv', sep='\t')
print(test.head())
