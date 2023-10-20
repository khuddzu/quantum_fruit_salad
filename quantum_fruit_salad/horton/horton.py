import torch
import h5py
import iodata
import grid.basegrid
from grid.becke import BeckeWeights
from grid.molgrid import MolGrid
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeRTransform
from gbasis.wrappers import from_iodata
from gbasis.evals.density import evaluate_density
import numpy as np
from denspart.mbis import MBISProModel
from denspart.vh import optimize_reduce_pro_model
from denspart.properties import compute_radial_moments, compute_multipole_moments, safe_ratio

def grid_and_density(fchk:str, 
        refinement:str, 
        rgrid: grid.basegrid.OneDGrid):
    mol = iodata.load_one(fchk)
    grid = MolGrid.from_preset(mol.atnums, mol.atcoords, rgrid, refinement, BeckeWeights())
    one_rdm = np.dot(mol.mo.coeffs * mol.mo.occs, mol.mo.coeffs.T)
    basis = from_iodata(mol)
    density = evaluate_density(one_rdm, basis[0], grid.points, coord_type=basis[1])
    return mol, grid, density

def compute_pro_model(mol: iodata.iodata.IOData,
        grid: MolGrid, 
        density: np.ndarray):
    pro_model_init = MBISProModel.from_geometry(mol.atnums, mol.atcoords)
    pro_model, localgrids = optimize_reduce_pro_model(pro_model_init,grid,density)
    return pro_model, localgrids

def pro_model_results(pro_model:MBISProModel, 
        localgrids: list, 
        grid: MolGrid,
        density: np.ndarray):
    results = pro_model.to_dict()
    if results['class']:
        del results['class']
    if results['propars'].any():
        del results['propars']
    results.update({"charges": pro_model.charges,
                 "multipole_moments": compute_multipole_moments(pro_model, grid, density, localgrids),
               "radial_moments": compute_radial_moments(pro_model, grid, density, localgrids)})
    return results

def update_multi_file_results(multi_results: dict, 
        single_results: dict):
    main_key = f'{len(single_results["atnums"]):03d}'
    if not multi_results:
        multi_results[main_key]={k:torch.tensor(v).unsqueeze(0) for k,v in single_results.items()}
    else:
        if main_key in multi_results:
            for key, value in single_results.items():
                if key in multi_results[main_key]:
                    if multi_results[main_key][key].ndim != torch.tensor(value).ndim:
                        multi_results[main_key][key] = torch.cat((multi_results[main_key][key], torch.tensor(value).unsqueeze(0)), dim=0)
                    else:
                        multi_results[main_key][key] = torch.cat((multi_results[main_key][key].unsqueeze(0), torch.tensor(value).unsqueeze(0)), dim=0)

        else:
            multi_results[main_key] = {k:torch.tensor(v).unsqueeze(0) for k,v in single_results.items()}
    return multi_results

def save_results_to_hdf5(out_file: str, global_results: dict):
    with h5py.File(out_file, 'w') as hf:
        for key, value in global_results.items():
            hf.create_group(key)
            for prop, v in value.items():
                hf[key].create_dataset(prop, data=v)
