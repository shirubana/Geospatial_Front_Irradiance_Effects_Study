#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pvdeg
import xarray as xr
import os
import glob

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# In[ ]:


files = glob.glob(os.path.abspath("/projects/pvsoiling/pvdeg/analysis/pvrw2025/diffuse_stow/tmy/*.nc"))

loaded_tmy = xr.open_mfdataset(files)


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.LambertConformal()})

# Plot using pvdeg's plot_sparse_analysis
fig, ax, im = pvdeg.geospatial.plot_sparse_analysis(
    loaded_tmy,
    "delta_mov (%)",
    resolution=400j,
    method="nearest",
    ax=ax,
    cmap="viridis_r"
)

# Set the title on the axis
ax.set_title("delta_mov (%)")

# Save the plot with high resolution
plt.savefig("delta_mov_percent_r.png", dpi=800)

# Show the plot
plt.show()


# In[ ]:


#fig, ax = plt.subplots(figsize=(12, 8))
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.LambertConformal()})

# Plot using pvdeg's plot_sparse_analysis
fig, ax, im = pvdeg.geospatial.plot_sparse_analysis(
    loaded_tmy,
    "energy_TT (Wh)",
    resolution=400j,
    method="nearest",
    ax=ax
)

# Set the title on the axis
ax.set_title("energy_TT (kWh)")

# Save the plot with high resolution
plt.savefig("energy_TT_kwh.png", dpi=800)

# Show the plot
plt.show()


# In[ ]:


loaded_tmy["


# In[ ]:





# In[2]:


five_min_loaded = xr.open_mfdataset("/projects/pvsoiling/pvdeg/analysis/pvrw2025/diffuse_stow/5min/*.nc")


# In[19]:


fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.LambertConformal()})

# Plot using pvdeg's plot_sparse_analysis
fig, ax, im = pvdeg.geospatial.plot_sparse_analysis(
    five_min_loaded,
    "delta_en (%)",
    resolution=200j,
    method="nearest",
    ax=ax,
    cmap="viridis"
)

# Set the title on the axis
ax.set_title("delta_mov (%)")

# Save the plot with high resolution
plt.savefig("delta_en_percent_5min_percent.png", dpi=800)

# Show the plot
plt.show()


# In[16]:


five_min_loaded


# In[ ]:




