import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy
import cartopy as cart
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


#================================================================================
def plot_field(field, lon=None, lat=None, num_contours=8, plot_lat_labels=True, plot_lon_labels=True, minC=None, maxC=None, cmap=plt.cm.bwr):

    Nlon,Nlat = np.shape(field) 
    if lon is None:
        lon = np.linspace(0,360,Nlon)
    if lat is None:
        lat = np.linspace(-90,90,Nlat)        

    field_plot = copy.deepcopy(field.T)
    print('MIN=',np.min(field_plot), ' MAX=',np.max(field_plot))
    if maxC!=None and minC!=None and minC<maxC: 
        field_plot = np.clip(field_plot,minC+np.abs(minC)*1e-2,maxC-np.abs(maxC)*1e-2)
        cb_lev = np.linspace(minC,maxC,num_contours)
        h = plt.contourf(lon,lat,field_plot,cb_lev,cmap=cmap, transform=ccrs.PlateCarree())
    else:
        minC = np.min(field_plot)
        maxC = np.max(field_plot)
        if minC<0.0 and maxC>0.0:
            maxC = max(-minC, maxC)*0.8
            minC = -maxC
            field_plot = np.clip(field_plot,minC+np.abs(minC)*1e-2,maxC-np.abs(maxC)*1e-2)
            cb_lev = np.linspace(minC,maxC,num_contours)
            h = plt.contourf(lon,lat,field_plot,cb_lev,cmap=cmap, transform=ccrs.PlateCarree())
        else:
            h = plt.contourf(lon,lat,field_plot,cmap=cmap, transform=ccrs.PlateCarree())
    
    return h

#================================================================================
def add_plot_features(fig, h, ax, xticks=[0, 60, 120, 180, 240, 300, 360], yticks=[-60, -30, 0, 30, 60], orientation='horizontal', tick_rotation=0, ymin=-90, ymax=90, borders=True, coasts=True):
    plt.ylim(ymin,ymax)
    if orientation is not None:
        cb = fig.colorbar(h, orientation=orientation)
        cb.ax.tick_params(labelsize='20')
        cb.ax.set_xticklabels(cb.ax.get_xticklabels(), rotation=tick_rotation)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    if coasts:
        ax.coastlines(linewidth=2)
    if borders:
        ax.add_feature(cart.feature.BORDERS, zorder=100, edgecolor='k')
    plt.grid()

#================================================================================
def zero_lower_diagonal(matrix):
    Z = copy.deepcopy(matrix)
    Tr = len(Z[0,:])
    for i in range(0,Tr):
        for j in range(i+1,Tr):
            Z[i,j] = None
    Z[0,0] = None
    return Z

#================================================================================
def plot_aniso_subgrid_coef(ax, Z, n, m, cbticks, xyticks, cbtick_fontsize=20, cmap=plt.cm.viridis, orientation='vertical', rotation=0):
    if cbticks is not None:
        epsilon = np.max(np.abs(cbticks))*1.0e-6
        plt.contourf(n, m, np.clip(Z,np.min(cbticks)+epsilon,np.max(cbticks)-epsilon), cbticks, cmap=cmap)
        if cbticks[0]*cbticks[-1]>0.0:
            cbar = plt.colorbar(ticks=cbticks, orientation=orientation)
        else:
            cbar = plt.colorbar(ticks=cbticks, orientation=orientation)
    else:
        plt.contourf(n, m, Z, cmap=cmap)
        cbar = plt.colorbar(orientation=orientation)

    cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=rotation)
    cbar.ax.tick_params(labelsize=cbtick_fontsize)
    
    plt.plot(m,n,'k-')
    plt.xlabel('$m$',labelpad=-5) ; plt.ylabel('$n$')
    if xyticks is not None:
        xt=ax.set_xticks(xyticks); yt=ax.set_yticks(xyticks)
    plt.grid()

#================================================================================
def plot_spectra_comparison(ax, DNS_n_L, LES_n_L, DNS_L, NoSGS_L, AS_L, AD_L, ASNoMean_L, ADNoMean_L, DNS_L_min, DNS_L_max,\
                            xmin=1, xmax=1.2e3, ymin=1e-12, ymax=1e-4, plot_range=False, scale_label_positions_by=1):
             
    plt.plot(DNS_n_L, DNS_L, 'r--', lw=2); 
    if plot_range:
        plt.fill_between(DNS_n_L, DNS_L_min, DNS_L_max, color='r', alpha=0.2); 
    plt.plot(LES_n_L, NoSGS_L, 'k-', lw=2); 

    plt.plot(DNS_n_L, DNS_L*1e-1, 'r--', lw=2); 
    if plot_range:
        plt.fill_between(DNS_n_L, DNS_L_min*1e-1, DNS_L_max*1e-1, color='r', alpha=0.2); 
    plt.plot(LES_n_L, AS_L*1e-1, 'k-', lw=2); 

    plt.plot(DNS_n_L, DNS_L*1e-2, 'r--', lw=2); 
    if plot_range:
        plt.fill_between(DNS_n_L, DNS_L_min*1e-2, DNS_L_max*1e-2, color='r', alpha=0.2); 
    plt.plot(LES_n_L, AD_L*1e-2, 'k-', lw=2); 

    plt.plot(DNS_n_L, DNS_L*1e-3, 'r--', lw=2); 
    if plot_range:
        plt.fill_between(DNS_n_L, DNS_L_min*1e-3, DNS_L_max*1e-3, color='r', alpha=0.2); 
    plt.plot(LES_n_L, ASNoMean_L*1e-3, 'k-', lw=2); 

    plt.plot(DNS_n_L, DNS_L*1e-4, 'r--', lw=2); 
    if plot_range:
        plt.fill_between(DNS_n_L, DNS_L_min*1e-4, DNS_L_max*1e-4, color='r', alpha=0.2); 
    plt.plot(LES_n_L, ADNoMean_L*1e-4, 'k-', lw=2); 

    plt.grid()
    plt.xlabel('$n$'); plt.xlim(xmin,xmax); plt.xscale('log'); 
    plt.yscale('log'); 
    
    if ymin is not None and ymax is not None:
        ax.text(500, scale_label_positions_by*ymax/1e3, 'none', fontsize=20)
        ax.text(500, scale_label_positions_by*ymax/1e4, 'S', fontsize=20)
        ax.text(500, scale_label_positions_by*ymax/1e5, 'D', fontsize=20)
        ax.text(500, scale_label_positions_by*ymax/1e6, 'S, $\\mathbf{\\bar{f}}=0$', fontsize=20)
        ax.text(500, scale_label_positions_by*ymax/1e7, 'D, $\\mathbf{\\bar{f}}=0$', fontsize=20)
        y_arrow = 20*ymin
        plt.ylim(ymin,ymax); 
        ax.annotate('$k_R$', (288, ymin),
                xytext=(288, y_arrow), 
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=24,
                horizontalalignment='right', verticalalignment='top')


