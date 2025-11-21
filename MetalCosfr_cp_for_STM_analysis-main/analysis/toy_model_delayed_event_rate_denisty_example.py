import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmology
from astropy.cosmology import z_at_value

# -------------------------------
# Utilities
# -------------------------------
def get_time_redshift_bins(z_start=10., z_stop=0., dt=60.):
    # time and redshift binning
    # dt=60. [Myr]
    if(z_stop==0.):
        cosmic_time = np.arange(cosmology.age(1e-7).to(u.Myr).value,\
             cosmology.age(z_start).to(u.Myr).value-dt, -dt )
    else:
        cosmic_time = np.arange(cosmology.age(z_stop).to(u.Myr).value,\
             cosmology.age(z_start).to(u.Myr).value-dt, -dt)
    redshifts = np.array([z_at_value(cosmology.age,  t*u.Myr) for t in cosmic_time])
    log1pz = np.array([np.log10(1.+zi) for zi in redshifts])
    lookback_time = cosmology.lookback_time(redshifts).to(u.Myr).value    
    return cosmic_time, redshifts, lookback_time, dt

def get_metallicity_bins(description=False, arrays=False):
    FOH_binning=['12+log(O/H)','5.3','9.7','200','GS98','8.83']
    FOH_binning_desc=['metallicity measure', 'min value', 'max value', 'number of bins', 'solar scale', 'solar ref. value']
    FeH_binning=['[Fe/H]','-4.3','2.3','200','GS98','7.5']
    FeH_binning_desc=['metallicity measure', 'min value', 'max value', 'number of bins', 'solar scale', 'solar ref. value (12+logFeH sun)']
    OFe_binning=['[O/Fe]','-0.3','0.8','200','GS98','1.33']
    OFe_binning_desc=['metallicity measure', 'min value', 'max value', 'number of bins', 'solar scale', 'solar ref. value (logOFe sun)']
    if(arrays): 
        FOH_min, FOH_max, nbins, FOHsun=\
            float(FOH_binning[FOH_binning_desc.index('min value')]),\
            float(FOH_binning[FOH_binning_desc.index('max value')]),\
            int(FOH_binning[FOH_binning_desc.index('number of bins')]),\
            float(FOH_binning[FOH_binning_desc.index('solar ref. value')])
        FOH_arr = np.linspace(FOH_min, FOH_max,nbins)
        
        FeH_min, FeH_max, nbins=\
            float(FeH_binning[FeH_binning_desc.index('min value')]),\
            float(FeH_binning[FeH_binning_desc.index('max value')]),\
            int(FeH_binning[FeH_binning_desc.index('number of bins')])
        FeH_arr = np.linspace(FeH_min, FeH_max,nbins)
        
        OFe_min, OFe_max, nbins=\
            float(OFe_binning[OFe_binning_desc.index('min value')]),\
            float(OFe_binning[OFe_binning_desc.index('max value')]),\
            int(OFe_binning[OFe_binning_desc.index('number of bins')])
        OFe_arr = np.linspace(OFe_min, OFe_max,nbins)
        return FOH_arr,FeH_arr,OFe_arr
    elif(description): return FOH_binning_desc,FeH_binning_desc,OFe_binning_desc
    else: return FOH_binning,FeH_binning,OFe_binning

def get_model_name(SFMR_ref='P23', calib='C20_ADF',\
     evolving_low_M_slope=True, OFe_ref='pl400CCFep1', FMR_slope=0.27, delta_FMR_ZOH_asym_0=0.,\
     SB_ref='Boco', IMF_ref='K01',lMmin=6., lMmax=12., z_start=10.,cluster_ref='None',\
     any_addons='None'):

    if(delta_FMR_ZOH_asym_0!=0.): MZR_evol=True
    else: MZR_evol=False
    
    model_name = 'SFR-'+str(SFMR_ref)+'-Z-'+str(calib)+'-Zev-'+str(MZR_evol)+'-GSMFev-'+str(evolving_low_M_slope)+'-OFe-'+str(OFe_ref)+'-FMR0-'+str(FMR_slope)[2:]+'-SB-'+str(SB_ref)
    
    if(delta_FMR_ZOH_asym_0!=0.): model_name+='-dFMR3_10-'+str(delta_FMR_ZOH_asym_0)[2:]
    if(IMF_ref!='K01'): model_name+='-IMF-'+str(IMF_ref)
    if(lMmin!=6.):
        model_name+='-lMmin-'+str(lMmin)
    if(lMmax!=12.):
        model_name+='-lMmax-'+str(lMmax) 
    if(z_start!=10.):
        model_name+='-z_start-'+str(z_start) 
    if(cluster_ref!='None'):
        model_name+='-clusters-'+str(cluster_ref)
    if(any_addons!='None'):
        model_name+='-add-'+str(any_addons)
    return model_name
    
def get_data_weight_by_metallicity_form_efficiency(
    metallicity_probe, model_name='', data_path='',\
    cosmic_time=[], redshifts=[], dt=60., 
    Xieff='none', Xeff_params_desc=[], Xeff_params=[],
    cluster_data=False, cluster_str='form', bin_val=10**5.,
    cluster_mass_binning=[10**3.5, 1e9, 13],
    cluster_mass_binning_desc=['min value', 'max value', 'number of bins']):
    """
    Reads redshift-and-metallicity-binned data for a given model, and optionally applies
    a weighting function (Xieff) to account for formation efficiency.

    Parameters
    ----------
    metallicity_probe : str
        Which metallicity axis to use ('FeH', 'OFe', 'OH', etc.)
    model_name : str
        model_ref name to read from.
    data_path : str
        Path to the model data.
    cosmic_time, redshifts : arrays (optional)
        If empty, will be generated internally using get_time_redshift_bins.
    dt : float
        Default time step in Myr.
    Xieff : str or callable
        If 'none', no weighting is applied. If callable, used as weighting function.
    Xeff_params_desc, Xeff_params : list
        Parameters for the efficiency function, interpreted by user-supplied callable.
    cluster_data : bool
        Whether to use cluster-based data.
    cluster_str : str
        Cluster suffix if cluster_data=True.
    bin_val : float
        Mass binning value for cluster data.
    cluster_mass_binning, cluster_mass_binning_desc : list
        Cluster mass binning parameters.

    Returns
    -------
    data : ndarray
        Weighted data array with shape (time, metallicity).
    mtot_z: data summed over metallicities at each z
    redshifts[::-1] : ndarray (from 0 to max)
    lbt: lookback time (from 0 to max) ndarray
    cosmic_time[::-1] : ndarray
    delt
    XH_arr : ndarray
        Metallicity bin edges/values used.
    """

    # Get cosmic time and redshift bins if not passed in
    if len(cosmic_time) == 0 or len(redshifts) == 0:
        cosmic_time, redshifts, _, dt = get_time_redshift_bins(z_start=10., z_stop=0., dt=dt)

    # Get metallicity binning
    FOH_binning, FeH_binning, OFe_binning = get_metallicity_bins()
    FOH_arr, FeH_arr, OFe_arr = get_metallicity_bins(arrays=True)

    # Determine which metallicity probe to use
    if metallicity_probe in ['OH', 'FOH', '12+log(O/H)', '[O/H]']:
        file_suffix = f"FOH_z_dM_clusters_{cluster_str}.dat" if cluster_data else "FOH_z_dM.dat"
        XH_arr = FOH_arr - 8.83
    elif metallicity_probe in ['FeH', 'FFeH', '12+log(Fe/H)', '[Fe/H]']:
        file_suffix = f"FeH_z_dM_clusters_{cluster_str}.dat" if cluster_data else "FFeH_z_dM.dat"
        XH_arr = FeH_arr
    elif metallicity_probe in ['OFe', '[O/Fe]']:
        file_suffix = f"OFe_z_dM_clusters_{cluster_str}.dat" if cluster_data else "OFe_z_dM.dat"
        XH_arr = OFe_arr
    else:
        raise ValueError("Invalid metallicity probe specified.")

    # Calculate time differences
    delt = [*[dt], *(np.abs(cosmic_time[1:] - cosmic_time[:-1]))]

    # Read data
    if cluster_data:
        cluster_mass_bins = get_cluster_mass_bins(cluster_mass_binning=cluster_mass_binning,
                                                  cluster_mass_binning_desc=cluster_mass_binning_desc)
        input_file = f"{data_path}{model_name}/{file_suffix}"
        data = read_data_for_cluster_mass_bin(file_path=input_file, bin_edges=cluster_mass_bins,
                                              value=bin_val, dt=dt)
        data = np.array(data)
    else:
        input_file = f"{data_path}{model_name}/{file_suffix}"
        with open(input_file, 'r') as f:
            data = np.array([[float(val) / (1e6 * dti) for val in line.split()]
                             for line, dti in zip(f.readlines(), delt)])

    # Apply weighting by formation efficiency if specified
    if Xieff != 'none' and Xieff !=None :
        if callable(Xieff):
            # user-supplied function
            filter_metallicity = np.zeros_like(data)
            for x, ii in zip(XH_arr, range(len(XH_arr))):
                filter_metallicity[:, ii] = Xieff(x, *Xeff_params)
            data *= filter_metallicity
        else:
            raise ValueError("Xieff must be 'none' or a callable function.")

    sfh = []
    for zi in range(len(redshifts)):
        SFRD_z = np.sum(data[zi][:])
        sfh.append(SFRD_z)
    # Convert cosmic time to look-back time
    lbt = cosmology.lookback_time(redshifts).to(u.Myr).value    
    return data, sfh, redshifts[::-1], lbt[::-1], cosmic_time[::-1], delt, XH_arr

def DTD_power_law(t, normalize=1., Tmin=3e7, slope=1.1, Tmax=1.4e10):
    """
    Delay Time Distribution: power law form.
    Times in [yr].
    """
    if t < Tmin or t > Tmax:
        return 0.0
    if slope == 1.:
        norm = normalize / (np.log(Tmax) - np.log(Tmin))
    else:
        norm = normalize * (1.-slope) / (Tmax**(1.-slope) - Tmin**(1.-slope))
    return norm * t**(-slope)


def DTD_cdf(t_array, DTD_func, **DTD_kwargs):
    """
    Compute cumulative distribution function of a user-provided DTD.
    """
    t_vals = np.array(t_array)
    dtd_vals = np.array([DTD_func(tt, **DTD_kwargs) for tt in t_vals])
    cdf_vals = cumulative_trapezoid(dtd_vals, t_vals, initial=0)
    return interp1d(t_vals, cdf_vals, bounds_error=False, fill_value=(0,1))


# -------------------------------
# Formation Efficiency
# -------------------------------
def Logistic_function(x, L, x0, k, C):
    # x0 -- turnover point ("metallicity threshold")
    # L -- plateau level
    # C -- the other end plateau level
    # k -- governs the steepness of the drop
    return C + (L - C) / (1. + np.exp(-k * (x - x0)))


# -------------------------------
def calculate_delayed_event_contribution_conv(
    sfrd_time, cosmic_times,
    DTD_func,  # user-supplied DTD
    DTD_kwargs={'slope':1.1, 'Tmin':4e8, 'normalize':1.0},
    per_Gpc3=True
    ):
    """
    Compute event rate density from SFRD and a user-provided DTD, 
    -- DTD should be normalized to unity between 0 and Hubble time.

    Parameters
    ----------
    sfrd_time : callable
        (progenitor) formation rate density as function of time [Myr].
    cosmic_times : ndarray
        Array of cosmic times [Myr].
    DTD_func : callable
        Delay time distribution function (in years).
    DTD_kwargs : dict
        Parameters for the DTD_func.
    per_Gpc3 : bool
        If True, return rate per Gpc^3 (*1e9 if sfrd is per Mpc^3).

    Returns
    -------
    rate_contribution : ndarray
        Event rate density as function of cosmic time.
    """

    # convert times to yr
    current_age=min(cosmic_times)
    times_yr=(cosmic_times-current_age)* 1e6
    dt_yr = np.gradient(times_yr)
    sfr_vals = sfrd_time(cosmic_times)

    # Build DTD kernel
    dtd_vals = np.array([DTD_func(tt, **DTD_kwargs) for tt in times_yr])

    # --- Convolve SFRD with DTD ---
    rate_contribution = np.convolve(sfr_vals, dtd_vals*dt_yr, mode='full')[:len(times_yr)]

    # Convert to per Gpc^3 if needed
    if per_Gpc3:
        rate_contribution *= 1e9

    return rate_contribution

def calculate_delayed_event_contribution(sfrd_time, cosmic_times,\
    DTD_func,  # user-supplied DTD
    DTD_kwargs={'slope':1.1, 'Tmin':4e8, 'normalize':1.0},per_Gpc3=True):

    """
    Compute event rate density from SFRD and a user-provided DTD, 
    -- DTD should be normalized to unity between 0 and Hubble time.

    Parameters
    ----------
    sfrd_time : callable
        (progenitor) formation rate density as function of time [Myr].
    cosmic_times : ndarray
        Array of cosmic times [Myr].
    DTD_func : callable
        Delay time distribution function (in years).
    DTD_kwargs : dict
        Parameters for the DTD_func.
    per_Gpc3 : bool
        If True, return rate per Gpc^3 (*1e9 if sfrd is per Mpc^3).

    Returns
    -------
    rate_contribution : ndarray
        Event rate density as function of cosmic time.
    """
    #cosmic_times in Myr,  convert times to yr
    hubble_time_yr = cosmology.age(0).to(u.yr).value
    fine_t = np.logspace(1, np.log10(hubble_time_yr), 2000)  # from 1 Myr to Hubble time
    cdf_t = DTD_cdf(fine_t, DTD_func, **DTD_kwargs)
    
    cSFRD=sfrd_time(cosmic_times)
    rate_contribution = np.zeros(cosmic_times.shape)  
    time_Myr=[]          
    ii=0
    for ii in range(len(cosmic_times)):
        #loop from early Universe to present
        #temporary arrays to fill in with the "future" event rate from current SFR
        future_events_from_current_SFR = np.zeros(cosmic_times.shape)
        current_age=cosmic_times[ii]*1e6 #[yr]
        if(ii==0):
            future_events_from_current_SFR[ii]=\
            cSFRD[ii]*(cdf_t((cosmic_times[1]-cosmic_times[0])*1e6))
        i=0
        while(i<len(cosmic_times)-ii-1):
            future_events_from_current_SFR[ii+i+1]=\
            cSFRD[ii]*(cdf_t(cosmic_times[ii+i+1]*1e6-current_age)-\
            cdf_t(cosmic_times[ii+i]*1e6-current_age))
            i+=1

        rate_contribution+=future_events_from_current_SFR
    if per_Gpc3:
        rate_contribution *= 1e9        
    return rate_contribution 


def plot_event_and_formation_rates(
    cosmic_times, lookback_times, redshifts,
    rate, formation_rate=None, sfh=None,
    xaxis_bottom="redshift", xaxis_top="lookback_time",
    include_formation=True, include_sfh=False,
    rescale_mode=None,log_rate=False,local_rate=10,\
    ticklabsize=12,myfontsize=12, c_rate='cornflowerblue',\
    label_model=None,fig=None,ax1=None # options: None, "local", "peak"
):
    """
    Plot delayed event rate contribution vs cosmic time / lookback time / redshift,
    with optional overlays of formation rate and SFH.

    Parameters
    ----------
    cosmic_times : ndarray
        Array of cosmic times [Myr].
    lookback_times : ndarray
        Array of lookback times [Myr].
    redshifts : ndarray
        Array of redshifts corresponding to cosmic_times.
    event_rate : callable
        Event rate density as a function of cosmic time.
    formation_rate : ndarray (optional)
        Formation rate density over time (same shape as event_rate).
    sfh : ndarray (optional)
        Star formation history over time (same shape).
    xaxis_bottom : str
        Choice for bottom x-axis: "cosmic_time", "lookback_time", "redshift".
    xaxis_top : str
        Choice for top x-axis: "cosmic_time", "lookback_time", "redshift".
    include_formation : bool
        Whether to include formation_rate in the plot.
    include_sfh : bool
        Whether to include sfh in the plot.
    rescale_mode : str or None
        How to rescale formation/sfh: 
        - None: no rescaling
        - "local": match value at z=0 (local Universe)
        - "peak": match the peak of the event rate
    """


    if(fig==None):
        fig, ax1 = plt.subplots(figsize=(8,5))
    top_axis,bottom_axis=xaxis_top,xaxis_bottom
    # Choose bottom x-axis
    if bottom_axis == "cosmic_time":
        x = cosmic_times
        xlabel = "Cosmic Time [Myr]"
    elif bottom_axis == "lookback_time":
        x = lookback_times/1e3
        xlabel = "Lookback Time [Gyr]"
    elif bottom_axis == "redshift":
        x = redshifts
        xlabel = "Redshift"
    else:
        raise ValueError("bottom_axis must be 'cosmic_time', 'lookback_time', or 'redshift'")

    event_rate=rate(cosmic_times)
    if local_rate != None:
            scale_factor = local_rate / event_rate[-1]
            event_rate*=scale_factor
    # Plot main curve
    if(log_rate): Y=np.log10(event_rate)
    else: Y=event_rate
    if(label_model==None):
        ax1.plot(x, Y, color=c_rate)
    else:
        ax1.plot(x, Y, label=label_model, color=c_rate)
    ax1.set_xlabel(xlabel,fontsize=myfontsize)
    if(log_rate):
        ax1.set_ylabel("log$_{10}$( Event Rate Density [Gpc$^{-3}$ yr${-1}$] )",fontsize=myfontsize)
    else:
        ax1.set_ylabel("Event Rate Density [Gpc$^{-3}$ yr${-1}$]",fontsize=myfontsize)

    def rescale(series, label):
        if series is None:
            return None, label
        if rescale_mode is None:
            return np.array(series), label
        if rescale_mode == "local":
            scale_factor = event_rate[-1] / series[-1]
            return np.array(series) * scale_factor, f"{label} (scaled to local)"
        if rescale_mode == "peak":
            scale_factor = np.max(event_rate) / np.max(series)
            return np.array(series) * scale_factor, f"{label} (scaled to peak)"
        raise ValueError("Invalid rescale_mode. Use None, 'local', or 'peak'.")

    if include_formation and formation_rate is not None:
        scaled_form, label_form = rescale(formation_rate, "Formation Rate")
        if(log_rate): Y=np.log10(scaled_form)
        else: Y=scaled_form
        ax1.plot(x, Y, "--", label=label_form, color=c_rate)

    if include_sfh and sfh is not None:
        scaled_sfh, label_sfh = rescale(sfh, "SFH")
        if(log_rate): Y=np.log10(scaled_sfh)
        else: Y=scaled_sfh        
        ax1.plot(x, Y, ":", label=label_sfh, lw=3,alpha=0.5,color="k")

    fig.canvas.draw()         
    xticks = ax1.get_xticks()
    xtick_labels = [tick.get_text() for tick in ax1.get_xticklabels()]
    secax = ax1.secondary_xaxis('top')
    yticks = ax1.get_yticks()   
#    ax.yaxis.tick_right() 
    ax1.yaxis.set_ticks_position('both')
    #secay = ax.secondary_yaxis('right')
    secax.set_xticks(xticks)
    top_labels = []
    for lbl in xtick_labels:
        if lbl:
            try:
                val = float(lbl)
                if(top_axis == "redshift"):
                    if(val==0 ): zval=0
                    elif(val>13.7): zval = z_at_value(cosmology.lookback_time, 13.7*u.Gyr).value
                    else: zval = z_at_value(cosmology.lookback_time, val*u.Gyr).value
                    #val = float(lbl.replace('−', '-'))  # replace Unicode minus
                    top_labels.append(f"{zval:.2g}")
                    secax.set_xlabel("redshift", fontsize=myfontsize)                      
                if(top_axis == "lookback_time"):
                    if(val==0): lbtval=0
                    else:
                        lbtval = cosmology.lookback_time(val).to(u.Gyr).value
                    top_labels.append(f"{lbtval:.3g}")
                    secax.set_xlabel("lookback time [Gyr]", fontsize=myfontsize)                  
            except ValueError:
                top_labels.append("")
        else:
            top_labels.append("")
    secax.set_xticklabels(top_labels)            
    secax.tick_params(axis='x', which='both', labelsize=ticklabsize)
    ax1.tick_params(axis='x', which='both', labelsize=ticklabsize)
    ax1.tick_params(axis='y', which='both', labelsize=ticklabsize)    
    plt.legend()
    plt.tight_layout()


def plot_event_rates_ratio(
    cosmic_times, lookback_times, redshifts,
    rate1, rate2,
    formation_rate1=None, formation_rate2=None,
    sfh=None,
    xaxis_bottom="redshift", xaxis_top="lookback_time",
    include_formation=False, include_sfh=False,
    rescale_mode=None, log_rate=False, local_rate=None,
    ticklabsize=12, myfontsize=12,
    c_rate="cornflowerblue", c_ratio="darkorange",
    label_model=None,label_formation=None,Ylabel=None, fig=None, ax1=None
):
    """
    Plot the ratio of two event rate densities, optionally including their
    formation rate ratios and SFH, with flexible x-axis choices.

    Parameters
    ----------
    cosmic_times, lookback_times, redshifts : ndarray
        Time/redshift arrays [Myr, Myr, dimensionless].
    rate1, rate2 : ndarray
        Event rates as a function of cosmic time.
    formation_rate1, formation_rate2 : ndarray (optional)
        Formation rate densities (same shape).
    sfh : ndarray (optional)
        Star formation history over time (same shape).
    xaxis_bottom, xaxis_top : str
        "cosmic_time", "lookback_time", or "redshift".
    include_formation : bool
        Whether to plot the ratio of formation rates.
    include_sfh : bool
        Whether to overlay SFH (single curve, not ratio).
    rescale_mode : str or None
        "local" (rescale to match z=0), "peak" (rescale to match peak),
        or None (no rescaling).
    log_rate : bool
        Plot log10 of ratios if True.
    local_rate : float or None
        If given, rescale event rates so that rate1(z=0) matches this value.
    """

    # --- Setup figure and axes
    if fig is None or ax1 is None:
        fig, ax1 = plt.subplots(figsize=(8, 5))

    # --- Select bottom x-axis
    if xaxis_bottom == "cosmic_time":
        x = cosmic_times
        xlabel = "Cosmic Time [Myr]"
    elif xaxis_bottom == "lookback_time":
        x = lookback_times / 1e3
        xlabel = "Lookback Time [Gyr]"
    elif xaxis_bottom == "redshift":
        x = redshifts
        xlabel = "Redshift"
    else:
        raise ValueError("xaxis_bottom must be 'cosmic_time', 'lookback_time', or 'redshift'")
        
    # --- Handle rate1 and rate2 as callable or array
    if callable(rate1):
        r1 = np.array(rate1(cosmic_times))
    else:
        r1 = np.array(rate1)

    if callable(rate2):
        r2 = np.array(rate2(cosmic_times))
    else:
        r2 = np.array(rate2)

    # --- Rescale rates if requested
    if local_rate is not None:
        scale_factor = local_rate / r1[-1]
        r1 *= scale_factor
        r2 *= scale_factor

    # --- Compute ratio
    ratio = np.divide(r1, r2, out=np.zeros_like(r1), where=(r2 != 0))

    # --- Plot ratio
    if log_rate:
        ax1.plot(x, np.log10(ratio), color=c_ratio, label="Rate1 / Rate2")
        ax1.set_ylabel("log$_{10}$(Rate1 / Rate2)", fontsize=myfontsize)
    else:
        if(label_model!=None): 
            ax1.plot(x, ratio, color=c_ratio, label=label_model)
        else:
            ax1.plot(x, ratio, color=c_ratio)
        if(Ylabel==None): ax1.set_ylabel("rate ratio", fontsize=myfontsize)
        else: ax1.set_ylabel(Ylabel, fontsize=myfontsize)

    # --- Optional: formation rate ratios
    if include_formation and (formation_rate1 is not None) and (formation_rate2 is not None):
        if callable(formation_rate1):
            f1 = np.array(formation_rate1(cosmic_times))
        else:
            f1 = np.array(formation_rate1)

        if callable(formation_rate2):
            f2 = np.array(formation_rate2(cosmic_times))
        else:
            f2 = np.array(formation_rate2)
    
        def rescale(series, ref, label):
            if rescale_mode is None:
                return series, label
            if rescale_mode == "local":
                return series * (ref[-1] / series[-1]), f"{label} (scaled local)"
            if rescale_mode == "peak":
                return series * (np.max(ref) / np.max(series)), f"{label} (scaled peak)"
            return series, label

        form_ratio = np.divide(f1, f2, out=np.zeros_like(f1), where=(f2 > 0))
        if(label_formation!=None): 
            ax1.plot(x, form_ratio, ls='--', color=c_ratio, label=label_formation)
        else:
            ax1.plot(x, form_ratio, ls='--', color=c_ratio,)        

    # --- Optional: SFH overlay
    if include_sfh and (sfh is not None):
        sfh_, _ = rescale(np.array(sfh), ratio, "SFH")
    
        ax1.plot(x, sfh_, ":", lw=2, alpha=0.6, color="k", label="SFH (scaled)")

    # --- Formatting
    ax1.set_xlabel(xlabel, fontsize=myfontsize)
    ax1.tick_params(axis="x", which="both", labelsize=ticklabsize)
    ax1.tick_params(axis="y", which="both", labelsize=ticklabsize)

    # --- Secondary x-axis (top)
    def convert_x(xvals, from_axis, to_axis):
        out = []
        for val in xvals:
            try:
                if to_axis == "redshift":
                    out.append(z_at_value(cosmology.lookback_time, val * u.Gyr).value
                               if val > 0 else 0.0)
                elif to_axis == "lookback_time":
                    out.append(cosmology.lookback_time(val).to(u.Gyr).value)
                elif to_axis == "cosmic_time":
                    out.append(cosmology.age(val).to(u.Gyr).value)
                else:
                    out.append(np.nan)
            except Exception:
                out.append(np.nan)
        return out

    secax = ax1.secondary_xaxis("top")
    xticks = ax1.get_xticks()
    if xaxis_bottom == "lookback_time":
        xticks_gyr = xticks
    elif xaxis_bottom == "cosmic_time":
        xticks_gyr = xticks / 1e3
    else:
        xticks_gyr = xticks

    if xaxis_top == "redshift":
        top_labels = [f"{z:.2g}" if z >= 0 else "" for z in convert_x(xticks_gyr, xaxis_bottom, "redshift")]
        secax.set_xlabel("Redshift", fontsize=myfontsize)
    elif xaxis_top == "lookback_time":
        top_labels = [f"{lt:.2f}" if lt >= 0 else "" for lt in convert_x(xticks_gyr, xaxis_bottom, "lookback_time")]
        secax.set_xlabel("Lookback Time [Gyr]", fontsize=myfontsize)
    elif xaxis_top == "cosmic_time":
        top_labels = [f"{ct:.2f}" if ct >= 0 else "" for ct in convert_x(xticks_gyr, xaxis_bottom, "cosmic_time")]
        secax.set_xlabel("Cosmic Time [Gyr]", fontsize=myfontsize)
    else:
        top_labels = ["" for _ in xticks]

    secax.set_xticks(xticks)
    secax.set_xticklabels(top_labels, fontsize=ticklabsize)

    ax1.legend()
    plt.tight_layout()
    return fig, ax1

# -------------------------------
# Example
# -------------------------------
if __name__ == "__main__":
    ''' 
    Example
    '''  
    plot_relative=False
    cosmic_times, redshifts, lookback_times, dt = get_time_redshift_bins()
    data_path='MetalCosfr_cp_for_STM_analysis-main/data/'

    SFMR_refs=["P23","P23modif_highz","P23slope08"]
    OFe_refs=["pl40CCFep03","pl400CCFep1","G05CCFep05","G05CCFep07"]
    OFe_color=['#e66101','#5e3c99','#01665e']
    
    #set model_ref by hand  or call get_model_name
    model_name='SFR-P23-Z-C20_ADF-Zev-True-GSMFev-True-OFe-pl40CCFep03-FMR0-27-SB-Boco-dFMR3_10-25'  
    #model_name2='SFR-P23-Z-C20_ADF-Zev-True-GSMFev-True-OFe-pl400CCFep1-FMR0-27-SB-Boco-dFMR3_10-25'  
    #model_name3='SFR-P23-Z-C20_ADF-Zev-True-GSMFev-True-OFe-G05CCFep05-FMR0-27-SB-Boco-dFMR3_10-25'  
    model_name = get_model_name(SFMR_ref=SFMR_refs[0], calib="C20_ADF", delta_FMR_ZOH_asym_0=0.25,\
         evolving_low_M_slope=True, OFe_ref=OFe_refs[0], FMR_slope=0.27,\
         SB_ref="Boco", IMF_ref='K01',lMmin=6., lMmax=12., z_start=10.,cluster_ref='None',\
         any_addons='None') 
    
    data_total, sfh_total = get_data_weight_by_metallicity_form_efficiency(
        metallicity_probe='FeH', model_name=model_name, data_path=data_path,\
        Xieff=None, Xeff_params_desc=[], Xeff_params=[])[0:2]

    L0=1.e-6 #note: this parameter sets the rate value
    Xeff_params=[L0,-1,-10.,L0*0.01]
    DTD_kwargs={'slope':1., 'Tmin':1e8, 'normalize':1.0}
    
    data_Xieff_weighted, sfh_FeH, _, _, cosmic_time, _, _ =\
    get_data_weight_by_metallicity_form_efficiency(
        metallicity_probe='FeH', model_name=model_name, data_path=data_path,\
        Xieff=Logistic_function, Xeff_params_desc=[], Xeff_params=Xeff_params)

    form_rate_density_time_FeH = interp1d(cosmic_time,\
        sfh_FeH, fill_value='extrapolate')


    if(False):
    # Compute event rate using convolution ---> 
    # slow with (unrealistically) dense bins needed for this calculation to be accurate,
    # to fix this one would need to sample t_del from DTD better in the regime where it varies the most
        cosmic_time=np.arange(min(cosmic_time), max(cosmic_time),5)
        event_rate = calculate_delayed_event_contribution_conv(
            sfrd_time=form_rate_density_time_FeH,
            cosmic_times=cosmic_time,
            DTD_func=DTD_power_law,
            DTD_kwargs=DTD_kwargs,per_Gpc3=True        
        )
        rate1 = interp1d(cosmic_time,\
            event_rate, fill_value='extrapolate')
        print('convolution method done -- needs dense time bins to be accurate')
    
    # Compute event rate; CDF method
    #cosmic_time=np.arange(min(cosmic_time), max(cosmic_time),60)
    event_rate = calculate_delayed_event_contribution(
        sfrd_time=form_rate_density_time_FeH,
        cosmic_times=cosmic_time,
        DTD_func=DTD_power_law,
        DTD_kwargs=DTD_kwargs,per_Gpc3=True        
    )
    rate = interp1d(cosmic_time,\
        event_rate, fill_value='extrapolate')
        
        
    if not plot_relative:
        fig, ax1 = plt.subplots(figsize=(8,5))
        ''' for plotting make sure to add [::-1] to time-redshift arrays 
            read via get_time_redshift_bins  '''
        include_formation,log_rate,local_rate=True,False,None
        plot_event_and_formation_rates(
            cosmic_times[::-1], lookback_times[::-1], redshifts[::-1],
            rate, formation_rate=sfh_FeH, sfh=sfh_total,
            xaxis_bottom="lookback_time", xaxis_top="redshift",
            include_formation=include_formation, include_sfh=True,
            rescale_mode='peak', log_rate=log_rate, local_rate=local_rate, c_rate='orange',\
            fig=fig, ax1= ax1,label_model='"metallicity"=Fe/H'
        )
    else:
        fig2, ax2 = plt.subplots(figsize=(8,5))
        include_formation=True 
        xaxis_bottom="lookback_time"#"redshift"
        xaxis_top="redshift"#"lookback_time"   
        include_sfh=False
        Ylabel='rate ratio (model 1/model 2)'
        
        plot_event_rates_ratio(
        cosmic_times, lookback_times, redshifts,
        rate1, rate_2,
        formation_rate1=form_rate_density_time_1, formation_rate2=form_rate_density_time_2,
        sfh=sfh_total[::-1],
        xaxis_bottom=xaxis_bottom, xaxis_top=xaxis_top,
        include_formation=include_formation, include_sfh=include_sfh,
        rescale_mode='peak', log_rate=False, local_rate=None,
        ticklabsize=12, myfontsize=12, Ylabel=Ylabel,
        c_rate="cornflowerblue", c_ratio="darkorange",
        label_model='event rate ratio ',\
        label_formation='formation rate ratio ',
        fig=fig2, ax1=ax2)
        
        ct=max(cosmic_times[::-1])
        print(rate(ct)/rate_OH(ct), max(rate(cosmic_times)/rate_OH(cosmic_times)))
    
plt.show()
