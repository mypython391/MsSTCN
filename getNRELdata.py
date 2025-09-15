#!/usr/bin/env python
'''
    Get wind speed data from NREL WIND
    https://www.nrel.gov/grid/wind-toolkit.html
    Select one wind farm with 100 turbines from Wyoming
'''
import h5pyd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs('data', exist_ok=True)

import h5pyd
# NREL HSDS endpoint + API key
# endpoint = "https://developer.nrel.gov/api/hsds"
# username = "t2111309925@163.com"                # literal string 'apikey'
# password = "aWlhvu3tfN3kLhF4rk3oE2KzNuz54OakC1PQvfze"   # replace with your actual key
#
# f = h5pyd.File("/nrel/wtk/conus/wtk_conus_2012.h5", 'r',
#                endpoint=endpoint, username=username, password=password)
# 设置HSDS服务的endpoint和API密钥
endpoint = "https://developer.nrel.gov/api/hsds"
api_key = "IVAqgc0OAVm0TyI8gON2edToj3zpALtGaLrCvI2b"  # 替换为你的NREL API密钥

# 打开WIND Toolkit数据文件
f = h5pyd.File("/nrel/wtk/conus/wtk_conus_2012.h5", 'r', endpoint=endpoint, api_key=api_key)
# f = h5pyd.File("/nrel/wtk/conus/wtk_conus_2012.h5", 'r')
meta = pd.DataFrame(f['meta'][()])

lon = -105.243988
lat = 41.868515
df = meta[(meta['longitude'] < lon+.25) & (meta['longitude'] >= lon)
          & (meta['latitude'] <= lat+.03) & (meta['latitude'] > lat-.18)]

df = df.drop([864121, 868456, 869542, 870629, 871718, 872807, 873897,
              876088, 866300, 867383, 868467, 869553, 870640])
df.to_csv('./data/wind_speed_meta.csv')
gid_list = list(df.index)
wind_speed_list = []
for gid in gid_list:
    wind_speed_list.append(f["windspeed_100m"][:, gid])

time_array = f['time_index'][()]
wind_speed_array = np.vstack(wind_speed_list)

wind_speed_df = pd.DataFrame(
    wind_speed_array, index=df.index, columns=time_array)
wind_speed_df = wind_speed_df/f['windspeed_100m'].attrs['scale_factor']

wind_speed_df.to_csv('./data/wind_speed.csv')
