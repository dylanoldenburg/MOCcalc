import numpy as np
import xarray as xr
import netCDF4 as netcdf
import matplotlib.pyplot as plt

def OHT_moc_calc(iso_thick,iso_vflux,iso_temp):

    ### params: all xarray dataarrays
    # iso_thick (isopycnal thickness - calculated using overturning_sigma_calc.py))
    # iso_vflux (volume flux in density space - calculated using overturning_sigma_calc.py)
    # iso_temp (temperature in density space - calculated using temp_sigma_coords.py)

    ### need parameters corresponding to the Fram and BSO sections ####
    data = np.load('/glade/work/oldend/CESM2_gridsec_fram_bso_params.npz')
    dx = data['dx']

    ### need zonal mean temperature and volume flux ####
    iso_temp_zonal = iso_temp*iso_thick*dx[np.newaxis,:,np.newaxis,np.newaxis]
    iso_temp_zonal = iso_temp_zonal.sum(dim="nvec")
    iso_temp_zonal = iso_temp_zonal/np.nansum(iso_thick*dx[np.newaxis,:,np.newaxis,np.newaxis],1)
    del iso_thick

    iso_vflux_zonal = iso_vflux.sum(dim="nvec")

    rho = 1027.5
    cw = 4186

    temp_overt = rho*cw*iso_vflux_zonal*iso_temp_zonal/1.e15
    oht_moc = temp_overt.sum(dim="sigma")

    return oht_moc
    
def OHT_moc_decomp_calc(iso_thick,iso_vflux,iso_temp):
    
    ### params: all xarray dataarrays
    # iso_thick (isopycnal thickness - calculated using overturning_sigma_calc.py))
    # iso_vflux (volume flux in density space - calculated using overturning_sigma_calc.py)
    # iso_temp (temperature in density space - calculated using temp_sigma_coords.py)

    ### need parameters corresponding to the Fram and BSO sections ####
    data = np.load('/glade/work/oldend/CESM2_gridsec_fram_bso_params.npz')
    dx = data['dx']

    iso_temp_zonal = iso_temp*iso_thick*dx[np.newaxis,:,np.newaxis,np.newaxis]
    iso_temp_zonal = iso_temp_zonal.sum(dim="nvec")
    iso_temp_zonal = iso_temp_zonal/np.nansum(iso_thick*dx[np.newaxis,:,np.newaxis,np.newaxis],1)
    del iso_thick
    iso_temp_mean = iso_temp_zonal[:732,:].mean(dim="time") ### taken relative to 1920-1980, hence use first 732 months
    iso_temp_anom = iso_temp_zonal-iso_temp_mean

    ##### make iso_vflux into xarray dataarray #####

    iso_vflux=xr.DataArray(iso_vflux,dims=iso_temp.dims,coords=iso_temp.coords,attrs={'long_name':'Volume flux','units':'m^3/s'})
    iso_vflux_zonal = iso_vflux.sum(dim="nvec")
    del iso_vflux, iso_temp
    iso_vflux_mean = iso_vflux_zonal[:732,:].mean(dim="time")
    iso_vflux_anom = iso_vflux_zonal-iso_vflux_mean

    rho = 1027.5
    cw = 4186

    temp_overt_mean = rho*cw*iso_vflux_mean*iso_temp_mean/1.e15
    temp_overt_active = rho*cw*iso_vflux_anom*iso_temp_mean/1.e15
    temp_overt_passive = rho*cw*iso_vflux_mean*iso_temp_anom/1.e15
    temp_overt_nonlinear = rho*cw*iso_vflux_anom*iso_temp_anom/1.e15

    oht_moc_passive = temp_overt_passive.sum(dim="nsig")
    oht_moc_active = temp_overt_active.sum(dim="nsig")
    oht_moc_nonlinear = temp_overt_nonlinear.sum(dim="nsig")
    oht_moc_mean = temp_overt_mean.sum(dim="nsig")

    return oht_moc_passive, oht_moc_active, oht_moc_nonlinear, oht_moc_mean
