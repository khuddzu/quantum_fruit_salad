import h5py
import torch


def combine_h5(combined_file: str, 
                dataset_1: str, 
                dataset_2: str):
    with h5py.File(combined_file, 'w') as out:
        with h5py.File(file1, 'r') as f1:
            with h5py.File(file2, 'r') as f2:
                for key in f1.keys():
                    d1 = f1[key]
                    out.create_group(key)

                    if key in f2:
                        d2 = f2[key]
                        for (k1, v1), (k2, v2) in tqdm.tqdm(zip(d1.items(), d2.items())):
                            assert k1 == k2
                            values = torch.cat((torch.tensor(v1), torch.tensor(v2)))
                            out[key].create_dataset(k1, data=values)
                    else:
                        for k,v in d1.items():
                            out[key].create_dataset(k, data=torch.tensor(v))
                        
                for key in f2.keys():
                    if key not in f1:
                        out.create_group(key)
                        d1 = f1[key]
                        d2 = f2[key]
                        for (k1, v1), (k2, v2) in tqdm(zip(d1.items(), d2.items())):
                            assert k1 == k2
                            values = torch.cat((torch.tensor(v1), torch.tensor(v2)))
                            out[key].create_dataset(k1, data=values)
        
