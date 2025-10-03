import h5py
import csv

filepath = "raw_data/genes.hdf5"
outpath = "data/genes.csv"
batch_size = 1000  # Adjust depending on memory

header = ["GeneID", "GeneSequence"]

with h5py.File(filepath, 'r') as f, open(outpath, "w", newline="") as o:
    writer = csv.DictWriter(o, fieldnames=header)
    writer.writeheader()
    
    batch = []
    for i, key in enumerate(f.keys()):
        # Convert bytes to string if needed
        seq = f[key][()]
        if isinstance(seq, bytes):
            seq = seq.decode("utf-8")
        batch.append({"GeneID": key, "GeneSequence": seq})
        
        if (i + 1) % batch_size == 0:
            writer.writerows(batch)
            batch = []
            print(f"Wrote {i + 1}/{len(f.keys())} genes")
    
    # Write remaining genes
    if batch:
        writer.writerows(batch)
        print(f"Wrote all {len(f.keys())} genes")