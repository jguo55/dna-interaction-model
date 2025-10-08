import csv
import h5py
import random
from sklearn.model_selection import train_test_split

def read_csvs(filepath='raw_data/'):
    dna_ids = []
    smiles_list = []
    labels = []

    with h5py.File(filepath + 'genes.hdf5', 'r') as f:
        keyset = set(f.keys())
        for i in range(1, 11):
            with open(filepath + f"ixns_file_{i}_of_10_with_SMILES.csv", 'r') as c:
                reader = csv.DictReader(c)
                for row in reader:
                    GeneID = str(int(float(row['GeneID'])))
                    if GeneID in keyset:
                        dna_ids.append(GeneID)
                        smiles_list.append(row['SMILES'])
                        labels.append(1)
            print(f"Loaded {len(dna_ids)} positive samples")
    return dna_ids, smiles_list, labels

def generate_negative_data(pos_ids, pos_smiles, multiplier=1): #generates negative data, for the model to learn better
    print("Generating Negative Pairs...")
    pos_pairs = set(zip(pos_ids, pos_smiles))
    seen = set()
    neg_dna_id = []
    neg_smiles = []
    neg_labels = []

    n = multiplier * len(pos_ids)

    while len(neg_dna_id) < n:
        dna_id = random.choice(pos_ids)
        smile = random.choice(pos_smiles)
        neg_pair = (dna_id, smile)
        if neg_pair not in pos_pairs and neg_pair not in seen:
            seen.add(neg_pair)
            neg_dna_id.append(dna_id)
            neg_smiles.append(smile)
            neg_labels.append(0)

    return neg_dna_id, neg_smiles, neg_labels

def write_csvs(dna_ids, smiles, labels, max_chunk_size=300000):
    if not (len(dna_ids) == len(smiles) == len(labels)):
        raise ValueError("dna_ids, smiles, and labels must have the same length")

    train_rows = []
    test_rows = []

    for dna, smile, label in zip(dna_ids, smiles, labels):
        if random.random() < 0.8:
            train_rows.append((dna, smile, label))
        else:
            test_rows.append((dna, smile, label))
    
    train_chunked = [train_rows[i:i+max_chunk_size] for i in range(0, len(train_rows), max_chunk_size)]
    test_chunked = [test_rows[i:i+max_chunk_size] for i in range(0, len(test_rows), max_chunk_size)]

    trainpath = "data_general/train/"
    testpath = "data_general/test/"

    for i, chunk in enumerate(train_chunked):
        with open(trainpath + f"train_{i+1}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("GeneID", "SMILES", "Label"))
            writer.writerows(chunk)

    for i, chunk in enumerate(test_chunked):
        with open(testpath + f"test_{i+1}.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(("GeneID", "SMILES", "Label"))
            writer.writerows(chunk)


if __name__ == "__main__":
    dna_ids, smiles_list, labels = read_csvs()
    neg_ids, neg_smiles, neg_labels = generate_negative_data(dna_ids, smiles_list)
    dna_ids.extend(neg_ids)
    smiles_list.extend(neg_smiles)
    labels.extend(neg_labels)
    write_csvs(dna_ids, smiles_list, labels)
    
    