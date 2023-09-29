import h5py
import torch


def combine_h5(combined_file: str, 
                dataset_1: str, 
                dataset_2: str):
    with h5py.File(intial_out, 'w') as out:
        with h5py.File(file1, 'r') as f1:
            with h5py.File(file2, 'r') as f2:
                shared=[]
                not_shared = []
                for key in f1.keys():
                    d1 = f1[key]
                    out.create_group(key)

                    if key in f2:
                        d2 = f2[key]
                        for (k1, v1), (k2, v2) in tqdm.tqdm(zip(d1.items(), d2.items())):
                            assert k1 == k2
                            cats = time.perf_counter()
                            values = torch.cat((torch.tensor(v1), torch.tensor(v2)))
                            cd = time.perf_counter()
                            out[key].create_dataset(k1, data=values)
                    else:
                        f1.copy(f1[key], out)
                for key in f2.keys():
                    if key not in f1:
                        f2.copy(f2[key], out)
            
