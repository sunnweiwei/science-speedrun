"""Chunked native-grid forecast output; unwritten predictions remain NaN."""
import numpy as np
import numcodecs
import xarray as xr
import zarr


def initialize(path, initialization, leads, members, levels, variables, latitude, longitude):
    case=np.asarray([initialization],dtype='datetime64[ns]')
    lead=np.asarray(leads,dtype='timedelta64[h]').astype('timedelta64[ns]')
    coords=dict(case=case,lead=lead,member=np.arange(members),level=levels,
                lat=latitude,lon=longitude,valid_time=(('case','lead'),case[:,None]+lead[None,:]))
    xr.Dataset(coords=coords).to_zarr(path,mode='w-',consolidated=False)
    group=zarr.open_group(str(path),mode='a')
    for name,spec in variables.items():
        dims=['case','lead','member']+(['level'] if spec['pressure_levels'] else [])+['lat','lon']
        shape=[1,len(leads),members]+([len(levels)] if spec['pressure_levels'] else [])+[len(latitude),len(longitude)]
        array=group.create_dataset(name,shape=shape,chunks=[1]*(len(shape)-2)+shape[-2:],
            dtype='float32',fill_value=np.nan,compressor=numcodecs.Blosc(cname='lz4',clevel=1,shuffle=numcodecs.Blosc.SHUFFLE))
        array.attrs.update(_ARRAY_DIMENSIONS=dims,units=spec['units'])
    zarr.consolidate_metadata(str(path))
    return group
