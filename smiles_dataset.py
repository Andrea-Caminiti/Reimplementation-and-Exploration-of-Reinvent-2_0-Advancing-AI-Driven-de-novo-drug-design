import numpy as np

if __name__ == '__main__':
    with open(r'data\chembl.mini.smi') as smi:
        lines = [line.strip() for line in smi.readlines()]

    res = np.random.choice(lines, 20_000, replace=False)
    
    with open(r'data\dataset.smi', 'w') as out:
        for line in res:
            out.write(line+'\n')