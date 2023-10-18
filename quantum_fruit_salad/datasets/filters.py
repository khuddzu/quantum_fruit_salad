import torch

def sort_data_by_xcoord(data):
    """
    Functions requires iterating through h5 file using h5py.
    Function takes a set of coordinates and sorts them based on atom with the smallest x-axis value.

    Example:

    Input: tensor([[[3, 5, 7],        Output: tensor([[[1, 4, 6],
                    [1, 4, 6]]])                       [3, 5, 7]]])

    This is done for all properties this molecule contains.
    Each property is sorted based on coordinate sorting indices.

    Input:
    data: HDF5 group; values from calling h5.items()

    Returns:
    data:
    """
    coordinates = torch.tensor(data['coordinates'][:])
    indices = coordinates[:, :, 0].sort()[1]
    new_data = {}
    criteria_shape = coordinates.shape[1]
    for key in data:
        prop = torch.tensor(data[key][:])
        if len(prop.shape) > 1:
            if prop.shape[1] == criteria_shape:
                new_prop = prop[torch.arange(prop.size(0)).unsqueeze(1), indices]
            else:
                new_prop=prop
        else:
            new_prop=prop
        new_data[key]=new_prop
    return new_data

def unique(x, dim=None):
    """Unique elements of x and indices of those unique elements

    Taken from following pytorch issue:
    https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810

    e.g.

    unique(tensor([
        [1, 2, 3],
        [1, 2, 4],
        [1, 2, 3],
        [1, 2, 5]
    ]), dim=0)
    => (tensor([[1, 2, 3],
                [1, 2, 4],
                [1, 2, 5]]),
        tensor([0, 1, 3]))
    """
    unique, inverse = torch.unique(
        x, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                        device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return unique, inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)

def high_force_magnitudes(data, 
                            threshold):
    for k,v in data.items():
        if 'force' in k.lower():
            if 'correction' in k.lower():
                continue
            elif 'noisy' in k.lower():
                continue
            else:
                print(k)
                forces = data[k][:]
                norm = torch.Tensor((torch.Tensor(forces).norm(dim=-1) > threshold).any(dim=-1).nonzero().squeeze(1))
                #if len(norm)>0:
                #    print(norm)
