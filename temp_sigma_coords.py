
""" Module that handles transport calculation across gridded sections """

# note: if heat transport: multiply by rho0_C
# if fwater: (salt - S0)/S0

import numpy as np
from xhistogram.xarray import histogram
import xarray as xr
import popmoc
import gsw

def temp_sigma_coords(temp, salt, meshfile):

    r""" Computation of overturning component:

        .. math::
            U^{ovt}(z) = \dfrac{\int_{l=0}^L U_{nonet}(z, l) dS(z, l)}{\int_{l=0}^L dS(z, l)}

    :param pypago.sections.GridSection section: Gridded section

    """
    ds = xr.open_dataset(meshfile)
    z_t = ds['z_t']/100.
    z_t = z_t.assign_coords(z_t=z_t)

    refz = 2000
    sigma_mid,sigma_edge = popmoc.sigma2_grid_86L()

    # Sigma2 on model TLAT, TLONG
    CT = gsw.conversions.CT_from_pt(salt,temp)
    sigma2_T = gsw.density.rho(salt,CT,refz)-1000
    sigma2_T = sigma2_T.assign_attrs({'long_name':'Sigma referenced to {}m'.format(refz),'units':'kg/m^3'})
    sigma2_T = sigma2_T.assign_coords(z_t=z_t)
    sigma2_T = sigma2_T.rename('sigma2_T')

    iso_count = histogram(sigma2_T, bins=[sigma_edge.values],dim=['z_t'],density=False)
    iso_count = iso_count.rename({'sigma2_T_bin':'sigma'}).assign_coords({'sigma':sigma_mid})

    # Use histogram to compute layer thickness. Vertical sum should be same as HT.
    dzwgts = (ds['dz']/100.).assign_attrs({'units':'m'})
    dzwgts = dzwgts.assign_coords(z_t=z_t)
    iso_thick = histogram(sigma2_T, bins=[sigma_edge.values], weights=dzwgts,dim=['z_t'],density=False)
    iso_thick = iso_thick.rename({'sigma2_T_bin':'sigma'}).assign_coords({'sigma':sigma_mid})
    iso_thick = iso_thick.rename('iso_thick').assign_attrs({'units':'m','long_name':'Isopycnal Layer Thickness'}).rename({'sigma':'sigma_mid'})
    iso_thick = iso_thick.transpose('time','nvec','sigma_mid')

    # Cumulative sum of layer thickness yields depth of layer edges:
    iso_depth = iso_thick.cumsum('sigma_mid').rename('iso_depth').rename({'sigma_mid':'sigma_bot'}).assign_attrs({'units':'m','long_name':'Isopycnal Layer Depth'})
    sigma_bot = sigma_edge.isel(sigma=slice(1,None)).rename({'sigma':'sigma_bot'}).assign_attrs({'long_name':'Sigma2 at bottom of layer'})
    iso_depth['sigma_bot'] = sigma_bot
    iso_depth = iso_depth.transpose('time','nvec','sigma_bot')

    temp = temp.where(temp<1.e10,0)
    # Volume fluxes in density-space. 
    iso_temp = histogram(sigma2_T, bins=[sigma_edge.values],weights=temp,dim=['z_t'],density=False)
    iso_temp = iso_temp.rename({'sigma2_T_bin':'sigma'}).assign_coords({'sigma':sigma_mid})
    iso_temp = iso_temp/iso_count
    return iso_temp, iso_thick, sigma_mid
