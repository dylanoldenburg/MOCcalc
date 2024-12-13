
""" Module that handles transport calculation across gridded sections """

# note: if heat transport: multiply by rho0_C
# if fwater: (salt - S0)/S0

import numpy as np
from xhistogram.xarray import histogram
import xarray as xr
import popmoc
import gsw

def overturning_sigma_calc(vel, temp, salt, meshfile):

    r""" Computation of overturning:
    
    Params:
    vel - velocity in m/s at desired sections
    temp - temperature in degC
    salt - absolute salinity
    meshfile - grid information for CESM2/POP (included here)

    """
    ds = xr.open_dataset(meshfile)
    z_t = ds['z_t']/100.
    z_t = z_t.assign_coords(z_t=z_t)

    fmoc = '/glade/u/home/yeager/analysis/python/POP_MOC/moc_template.nc'
    ds_moctemp = xr.open_dataset(fmoc)
    refz = 2000
    sigma_mid,sigma_edge = popmoc.sigma2_grid_86L()

    # Sigma2 on model TLAT, TLONG
    CT = gsw.conversions.CT_from_pt(salt,temp)
    sigma2_T = gsw.density.rho(salt,CT,refz)-1000
    sigma2_T = sigma2_T.assign_attrs({'long_name':'Sigma referenced to {}m'.format(refz),'units':'kg/m^3'})
    sigma2_T = sigma2_T.assign_coords(z_t=z_t)

    # Here, test histogram by counting cells in each density bin. Vertical sum should be same as KMT.
    iso_count = histogram(sigma2_T, bins=[sigma_edge.values],dim=['z_t'],density=False)
    iso_count = iso_count.rename({'salt_bin':'sigma'}).assign_coords({'sigma':sigma_mid})

    #kmtdiff = iso_count.sum('sigma') - ds['KMT']
    #print("Max difference from true KMT = {}".format(abs(kmtdiff).max().values))

    # Use histogram to compute layer thickness. Vertical sum should be same as HT.
    dzwgts = (ds['dz']/100.).assign_attrs({'units':'m'})
    dzwgts = dzwgts.assign_coords(z_t=z_t)
    iso_thick = histogram(sigma2_T, bins=[sigma_edge.values], weights=dzwgts,dim=['z_t'],density=False)
    iso_thick = iso_thick.rename({'salt_bin':'sigma'}).assign_coords({'sigma':sigma_mid})
    iso_thick = iso_thick.rename('iso_thick').assign_attrs({'units':'m','long_name':'Isopycnal Layer Thickness'}).rename({'sigma':'sigma_mid'})
    iso_thick = iso_thick.transpose('time','sigma_mid','nvec')

    #htdiff = iso_thick.sum('sigma_mid') - (ds['HT']/100.).assign_attrs({'units':'m'})
    #print("Max difference from true HT = {}m".format(abs(htdiff).max().values))

    # Cumulative sum of layer thickness yields depth of layer edges:
    iso_depth = iso_thick.cumsum('sigma_mid').rename('iso_depth').rename({'sigma_mid':'sigma_bot'}).assign_attrs({'units':'m','long_name':'Isopycnal Layer Depth'})
    sigma_bot = sigma_edge.isel(sigma=slice(1,None)).rename({'sigma':'sigma_bot'}).assign_attrs({'long_name':'Sigma2 at bottom of layer'})
    iso_depth['sigma_bot'] = sigma_bot
    iso_depth = iso_depth.transpose('time','sigma_bot','nvec')

    # Isopycnal depth of bottom-most layer should be same as HT.
    #htdiff =  iso_depth.isel(sigma_bot=-1) - (ds['HT']/100.).assign_attrs({'units':'m'})
    #print("Max difference from true HT = {}m".format(abs(htdiff).max().values))

    vel = vel.where(vel<1.e30,0)

    # Volume fluxes in density-space. 
    iso_vflux = histogram(sigma2_T, bins=[sigma_edge.values],weights=vel,dim=['z_t'],density=False)
    iso_vflux = iso_vflux.rename({'salt_bin':'sigma'}).assign_coords({'sigma':sigma_mid})

    # Vertical sum in density-space should reproduce vertical sum in depth-space
    #vfluxdiff = iso_vflux.sum('sigma') - vel.sum('z_t')
    #print("Max difference from true Vflux = {}".format(abs(vfluxdiff).max().values))

    # add vflux at southern boundary of Atlantic domain
    tmp = iso_vflux.sum('nvec')
    moc_s = -tmp.sortby('sigma',ascending=False).cumsum('sigma').sortby('sigma',ascending=True)/1.e6
    moc_s['sigma'] = sigma_edge.isel(sigma=slice(0,-1))

    return moc_s, iso_vflux, iso_thick, sigma_mid

def overturning_sigma_rev_0m(vel, temp, salt, meshfile):

    r""" Computation of overturning component:

        .. math::
            U^{ovt}(z) = \dfrac{\int_{l=0}^L U_{nonet}(z, l) dS(z, l)}{\int_{l=0}^L dS(z, l)}

    :param pypago.sections.GridSection section: Gridded section
    :param str velname: Name of the velocity field

    """
    ds = xr.open_dataset(meshfile)
    z_t = ds['z_t']/100.
    z_t = z_t.assign_coords(z_t=z_t)

    fmoc = '/glade/u/home/yeager/analysis/python/POP_MOC/moc_template.nc'
    ds_moctemp = xr.open_dataset(fmoc)
    refz = 0
    sigma_mid,sigma_edge = popmoc.sigma0_grid_86L()

    # Sigma2 on model TLAT, TLONG
    CT = gsw.conversions.CT_from_pt(salt,temp)
    sigma0_T = gsw.density.rho(salt,CT,refz)-1000
    sigma0_T = sigma0_T.assign_attrs({'long_name':'Sigma referenced to {}m'.format(refz),'units':'kg/m^3'})
    sigma0_T = sigma0_T.assign_coords(z_t=z_t)

    # Here, test histogram by counting cells in each density bin. Vertical sum should be same as KMT.
    #iso_count = histogram(sigma0_T, bins=[sigma_edge.values],dim=['z_t'],density=False)
    #iso_count = iso_count.rename({'salt_bin':'sigma'}).assign_coords({'sigma':sigma_mid})

    #kmtdiff = iso_count.sum('sigma') - ds['KMT']
    #print("Max difference from true KMT = {}".format(abs(kmtdiff).max().values))

    # Use histogram to compute layer thickness. Vertical sum should be same as HT.
    dzwgts = (ds['dz']/100.).assign_attrs({'units':'m'})
    dzwgts = dzwgts.assign_coords(z_t=z_t)
    iso_thick = histogram(sigma0_T, bins=[sigma_edge.values], weights=dzwgts,dim=['z_t'],density=False)
    iso_thick = iso_thick.rename({'salt_bin':'sigma'}).assign_coords({'sigma':sigma_mid})
    iso_thick = iso_thick.rename('iso_thick').assign_attrs({'units':'m','long_name':'Isopycnal Layer Thickness'}).rename({'sigma':'sigma_mid'})
    iso_thick = iso_thick.transpose('time','sigma_mid','nvec')

    #htdiff = iso_thick.sum('sigma_mid') - (ds['HT']/100.).assign_attrs({'units':'m'})
    #print("Max difference from true HT = {}m".format(abs(htdiff).max().values))

    # Cumulative sum of layer thickness yields depth of layer edges:
    iso_depth = iso_thick.cumsum('sigma_mid').rename('iso_depth').rename({'sigma_mid':'sigma_bot'}).assign_attrs({'units':'m','long_name':'Isopycnal Layer Depth'})
    sigma_bot = sigma_edge.isel(sigma=slice(1,None)).rename({'sigma':'sigma_bot'}).assign_attrs({'long_name':'Sigma2 at bottom of layer'})
    iso_depth['sigma_bot'] = sigma_bot
    iso_depth = iso_depth.transpose('time','sigma_bot','nvec')

    # Isopycnal depth of bottom-most layer should be same as HT.
    #htdiff =  iso_depth.isel(sigma_bot=-1) - (ds['HT']/100.).assign_attrs({'units':'m'})
    #print("Max difference from true HT = {}m".format(abs(htdiff).max().values))

    vel = vel.where(vel<1.e30,0)

    # Volume fluxes in density-space. 
    iso_vflux = histogram(sigma0_T, bins=[sigma_edge.values],weights=vel,dim=['z_t'],density=False)
    iso_vflux = iso_vflux.rename({'salt_bin':'sigma'}).assign_coords({'sigma':sigma_mid})

    # Vertical sum in density-space should reproduce vertical sum in depth-space
    #vfluxdiff = iso_vflux.sum('sigma') - vel.sum('z_t')
    #print("Max difference from true Vflux = {}".format(abs(vfluxdiff).max().values))

    # add vflux at southern boundary of Atlantic domain
    tmp = iso_vflux.sum('nvec')
    moc_s = tmp.sortby('sigma',ascending=True).cumsum('sigma').sortby('sigma',ascending=True)/1.e6
    moc_s['sigma'] = sigma_edge.isel(sigma=slice(0,-1))

    return moc_s, iso_vflux, sigma_mid

def overturning_sigma_calc_nemo(vel, temp, salt, meshfile):

    r""" Computation of overturning component:

        .. math::
            U^{ovt}(z) = \dfrac{\int_{l=0}^L U_{nonet}(z, l) dS(z, l)}{\int_{l=0}^L dS(z, l)}

    :param pypago.sections.GridSection section: Gridded section
    :param str velname: Name of the velocity field

    """
    ds = xr.open_dataset(meshfile)
    z = ds['gdept_1d'][0,:]
    z = z.assign_coords(z=z.values)
    refz = 2000
    sigma_mid,sigma_edge = popmoc.sigma2_grid_86L()

    # Sigma2 on model TLAT, TLONG
    CT = gsw.conversions.CT_from_pt(salt,temp)
    sigma2_T = gsw.density.rho(salt,CT,refz)-1000
    sigma2_T = sigma2_T.assign_attrs({'long_name':'Sigma referenced to {}m'.format(refz),'units':'kg/m^3'})
    sigma2_T = sigma2_T.assign_coords(z=z.values)

    # Here, test histogram by counting cells in each density bin. Vertical sum should be same as mbathy.
    #iso_count = histogram(sigma2_T, bins=[sigma_edge.values],dim=['z'],density=False)
    #iso_count = iso_count.rename({'salt_bin':'sigma'}).assign_coords({'sigma':sigma_mid})

    #mbathydiff = iso_count.sum('sigma') - ds['mbathy'][0,:]
    #print("Max difference from true mbathy = {}".format(abs(mbathydiff).max().values))

    # Use histogram to compute layer thickness. Vertical sum should be same as HT.
    dzwgts = (ds['e3t_1d'][0,:]).assign_attrs({'units':'m'})
    dzwgts = dzwgts.assign_coords(z=z.values)
    iso_thick = histogram(sigma2_T, bins=[sigma_edge.values], weights=dzwgts,dim=['z'],density=False)
    iso_thick = iso_thick.rename({'salt_bin':'sigma'}).assign_coords({'sigma':sigma_mid})
    iso_thick = iso_thick.rename('iso_thick').assign_attrs({'units':'m','long_name':'Isopycnal Layer Thickness'}).rename({'sigma':'sigma_mid'})
    iso_thick = iso_thick.transpose('time','sigma_mid','nvec')

    #htdiff = iso_thick.sum('sigma_mid') - (ds['HT']).assign_attrs({'units':'m'})
    #print("Max difference from true HT = {}m".format(abs(htdiff).max().values))

    # Cumulative sum of layer thickness yields depth of layer edges:
    iso_depth = iso_thick.cumsum('sigma_mid').rename('iso_depth').rename({'sigma_mid':'sigma_bot'}).assign_attrs({'units':'m','long_name':'Isopycnal Layer Depth'})
    sigma_bot = sigma_edge.isel(sigma=slice(1,None)).rename({'sigma':'sigma_bot'}).assign_attrs({'long_name':'Sigma2 at bottom of layer'})
    iso_depth['sigma_bot'] = sigma_bot
    iso_depth = iso_depth.transpose('time','sigma_bot','nvec')

    # Isopycnal depth of bottom-most layer should be same as HT.
    #htdiff =  iso_depth.isel(sigma_bot=-1) - (ds['HT']).assign_attrs({'units':'m'})
    #print("Max difference from true HT = {}m".format(abs(htdiff).max().values))

    vel = vel.where(vel<1.e30,0)

    # Volume fluxes in density-space. 
    iso_vflux = histogram(sigma2_T, bins=[sigma_edge.values],weights=vel,dim=['z'],density=False)
    iso_vflux = iso_vflux.rename({'salt_bin':'sigma'}).assign_coords({'sigma':sigma_mid})

    # Vertical sum in density-space should reproduce vertical sum in depth-space
    #vfluxdiff = iso_vflux.sum('sigma') - vel.sum('z')
    #print("Max difference from true Vflux = {}".format(abs(vfluxdiff).max().values))

    # add vflux at southern boundary of Atlantic domain
    tmp = iso_vflux.sum('nvec')
    moc_s = -tmp.sortby('sigma',ascending=False).cumsum('sigma').sortby('sigma',ascending=True)/1.e6
    moc_s['sigma'] = sigma_edge.isel(sigma=slice(0,-1))

    return moc_s, iso_vflux, sigma_mid
