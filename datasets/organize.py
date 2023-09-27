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
    coordinates = torch.tensor(data['coordinates'][:][0]).unsqueeze(0)
    indices = coordinates[:, :, 0].sort()[1]
    for prop in data:
        data[prop] = torch.tensor(data[prop][:][0])
        data[prop] = data[prop][torch.arange(data[prop].size(0)).unsqueeze(1), indices]
    return data
