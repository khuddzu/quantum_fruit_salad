import h5py
import torch
import tqdm 
import numpy as np 

def combine_h5(combined_file: str, 
                dataset_1: str, 
                dataset_2: str):
    with h5py.File(combined_file, 'w') as out:
        with h5py.File(dataset_1, 'r') as f1:
            with h5py.File(dataset_2, 'r') as f2:
                for key in f1.keys():
                    d1 = f1[key]
                    if key in f2:
                        out.create_group(key)
                        d2 = f2[key]
                        for (k1, v1), (k2, v2) in tqdm.tqdm(zip(d1.items(), d2.items())):
                            assert k1 == k2
                            try:
                                values = torch.cat((torch.tensor(np.array(v1)), torch.tensor(np.array(v2))))
                                out[key].create_dataset(k1, data=values)
                            except:
                                print(f'Key:{k1} could not be copied. dtype:{v1.dtype}')
                    else:
                        f1.copy(f1[key], out)
                for key in f2.keys():
                    if key not in f1:
                        f2.copy(f2[key], out)
            
