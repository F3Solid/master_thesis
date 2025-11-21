import numpy as np
import astropy.units as u
from astropy.cosmology import WMAP9 as cosmology
#https://docs.astropy.org/en/stable/cosmology/index.html
from astropy.cosmology import z_at_value
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import re

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def smooth(d, c=2):
    if c == 0:
        return d
    x = np.zeros(len(d))
    x[0] = (d[1]+d[0])/2
    x[-1] = (d[-1]+d[-2])/2
    x[1:-1] = (2*d[1:-1]+d[:-2]+d[2:])/4
    return smooth(x, c=c-1)
    
def get_time_redshift_bins(z_start=10., z_stop=0., dt=60.):
    #time and redshift binning
    #dt=60. #[Myr]
    #z_stop=0.
    #Use bins spaced equally in time, calculate redshift bins given the adopted cosmology
    if(z_stop==0.):
	    cosmic_time = np.arange(cosmology.age(1e-7).to(u.Myr).value,\
	     cosmology.age(z_start).to(u.Myr).value-dt, -dt )
    else:
	    cosmic_time = np.arange(cosmology.age(z_stop).to(u.Myr).value,\
	     cosmology.age(z_start).to(u.Myr).value-dt, -dt)
    redshifts = np.array([z_at_value(cosmology.age,  t*u.Myr) for t in cosmic_time])
    log1pz = np.array([np.log10(1.+zi) for zi in redshifts])
    return cosmic_time, redshifts, log1pz, dt
 
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


def float_to_p_string(value):
        if value.is_integer():
            return str(int(value))
        else:
            return str(value).replace('.', 'p')
            
def p_to_float(p_string):
        return float(p_string.replace('p', '.'))
        
def create_cluster_ref(slope_cluster_MF=2, lMmin_cluster=3.5, nbin_cluster=13,\
    lMmax_cluster=9, Mclmax_SFR_ref=None, frac_SE=None, dissolution_ref=None):
    
    # Convert input parameters to appropriate string format
    slope_str = float_to_p_string(float(slope_cluster_MF))
    lMmin_str = float_to_p_string(float(lMmin_cluster))
    lMmax_str = float_to_p_string(float(lMmax_cluster))    
    nbin_str = str(int(nbin_cluster))
    
    # Construct the cluster_ref string
    cluster_ref = f'CMFslope-{slope_str}-lMcmin-{lMmin_str}-nb-{nbin_str}'
    if(lMmax_cluster!=9): 
        cluster_ref=f'CMFslope-{slope_str}-lMcmin-{lMmin_str}-lMcmax-{lMmax_str}-nb-{nbin_str}'
    
    if(frac_SE is not None):
        cluster_ref+='-fracSE-'+float_to_p_string(float(frac_SE)) 
    if(Mclmax_SFR_ref is not None):
        cluster_ref+='-Mclmax_SFR_ref-'+Mclmax_SFR_ref
    if(dissolution_ref is not None):
        cluster_ref+='-dissolution_ref-'+dissolution_ref
    return cluster_ref

def extract_cluster_info(cluster_ref, default_slope=2, default_lMmin=3.5, default_nbin=13,default_lMmax=9,\
    default_frac_SE=0.8, default_Mclmax_SFR_ref='1e8at1e2',default_dissolution_ref='const_t4_5000'):

    # Define the default values
    slope_cluster_MF = default_slope
    lMmin_cluster = default_lMmin
    lMmax_cluster=default_lMmax
    nbin_cluster = default_nbin
    Mclmax_SFR_ref = default_Mclmax_SFR_ref
    frac_SE = default_frac_SE
    dissolution_ref=default_dissolution_ref
    
    # Regular expressions to extract the values from the string
    slope_pattern = r'CMFslope-(?P<slope>\d+p?\d*)'
    lMmin_pattern = r'lMcmin-(?P<lMmin>\d+p?\d*)'
    lMmax_pattern = r'lMcmax-(?P<lMmax>\d+p?\d*)'    
    nbin_pattern = r'nb-(?P<nbin>\d+)'
    frac_SE_pattern = r'fracSE-(?P<fracSE>\d+p?\d*)'
    Mclmax_SFR_pattern = r'Mclmax_SFR_ref-(?P<MmaxSFRref>)'

    # Search for the patterns in the string
    slope_match = re.search(slope_pattern, cluster_ref)
    lMmin_match = re.search(lMmin_pattern, cluster_ref)
    lMmax_match = re.search(lMmax_pattern, cluster_ref)    
    nbin_match = re.search(nbin_pattern, cluster_ref)
    frac_SE_match = re.search(frac_SE_pattern, cluster_ref)
    Mclmax_SFR_match = re.search(Mclmax_SFR_pattern, cluster_ref)
    # Extract and update the values if found
    if slope_match:
        slope_cluster_MF = p_to_float(slope_match.group('slope'))
    if lMmin_match:
        lMmin_cluster = p_to_float(lMmin_match.group('lMmin'))
    if lMmax_match:
        lMmax_cluster = p_to_float(lMmax_match.group('lMmax'))
    if nbin_match:
        nbin_cluster = int(nbin_match.group('nbin'))
    if frac_SE_match:
        frac_SE = p_to_float(frac_SE_match.group('fracSE'))

    parts = cluster_ref.split('-')
    for i in range(0, len(parts), 2):
        if parts[i] == 'Mclmax_SFR_ref':
            Mclmax_SFR_ref=parts[i+1]
        if parts[i] == 'dissolution_ref':
            dissolution_ref=parts[i+1]
                
    return slope_cluster_MF, lMmin_cluster,lMmax_cluster, nbin_cluster, frac_SE, Mclmax_SFR_ref,dissolution_ref

def get_cluster_mass_bins(cluster_mass_binning=[10**3.5,1e9,13],\
    cluster_mass_binning_desc=['min value', 'max value','number of bins']):
    #    [10**3.,1e9,13]
    cluster_mass_bins =\
    np.logspace(np.log10(float(cluster_mass_binning[cluster_mass_binning_desc.index('min value')])),\
    np.log10(float(cluster_mass_binning[cluster_mass_binning_desc.index('max value')])),\
    int(cluster_mass_binning[cluster_mass_binning_desc.index('number of bins')]))
    return cluster_mass_bins
    

def find_bin_index(bin_edges, value):
    """
    Finds the bin index for a given value based on the bin edges.
    
    :param bin_edges: List of bin edges
    :param value: Value of interest
    :return: Index of the bin where the value belongs, or -1 if out of bounds
    """    

    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= value < bin_edges[i + 1]:
            return i
    if value == bin_edges[-1]:  # Handle the edge case where value is exactly the last bin edge
        return len(bin_edges) - 2
    return -1  # Return -1 if the value does not fall within any bin

    
def Logistic_function(x,L,x0,k,C):
    #x0 -- turnover point ("metallicity threshold")
    #L -- plateau level
    #C -- the other end plateau level
    #k -- governs the steepnes of the drop
    return C+(L-C)/(1.+np.exp(-k*(x-x0)))
    
def get_colors_from_colormap(colormap_name='cividis', num_colors=10,low=0,high=1):
    # Get the colormap
    cmap = plt.get_cmap(colormap_name)
    
    # Generate an array of evenly spaced values between 0 and 1
    indices = np.linspace(low, high, num_colors)
    
    # Generate the colors
    colors = [cmap(i) for i in indices]
    
    return colors 


def read_data_for_cluster_mass_bin(file_path, bin_edges, value, dt=60.):
    """
    Reads every n-th line from the given file, where n is determined based on the selected bin index.
    
    :param file_path: Path to the input file
    :param bin_edges: List of bin edges
    :param value: Value to determine the bin index
    :return: A list containing every n-th line from the file starting from the selected bin index
    """
    def find_bin_index(bin_edges, value):
        """
        Finds the bin index for a given value based on the bin edges.
        
        :param bin_edges: List of bin edges
        :param value: Value of interest
        :return: Index of the bin where the value belongs, or -1 if out of bounds
        """
        for i in range(len(bin_edges) - 1):
            if bin_edges[i] <= value < bin_edges[i + 1]:
                return i
        if value == bin_edges[-1]:  # Handle the edge case where value is exactly the last bin edge
            return len(bin_edges) - 2
        return -1  # Return -1 if the value does not fall within any bin

    lines = []
    bin_idx = find_bin_index(bin_edges, value)
    if bin_idx == -1:
        print("Value is out of the bin range.")
        return lines

    n = len(bin_edges) - 1
    start_line = bin_idx - 1
    #print(n, start_line, bin_idx)
    
    try:
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if i >= start_line and (i - start_line) % n == 0:
                    lines.append([ float(data)/(1e6*dt) for data in line.split()])
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return lines
    
#def get_data_cut_in_metallicity(metallicity_probe='FeH',XH_lower_cut=-5,XH_upper_cut=5,\
#    model_name='',data_path='',cosmic_time=[],redshifts=[],dt=60.,info_print=False,smmothit=True,\
#    Xieff='none',Xeff_params_desc=[],Xeff_params=[],\
#    cluster_data=False ,cluster_str='form',bin_val=10**5.,\
#    cluster_mass_binning=[10**3.5,1e9,13],\
#    cluster_mass_binning_desc=['min value', 'max value','number of bins']):
#    #XH_lower_cut=-5,XH_upper_cut=5 i.e. NO CUT, total SFRD
#    # XH -- metallicity in log(X/H) - log(X/H)|solar units where XH=FeH or XH=OH or XH=OFe
#    FOH_binning,FeH_binning,OFe_binning=get_metallicity_bins()
#    FOH_binning_desc,FeH_binning_desc,OFe_binning_desc=get_metallicity_bins(description=True)  
#    FOH_arr,FeH_arr,OFe_arr=get_metallicity_bins(arrays=True)
#    
#    delt=[*[dt],*(np.abs(cosmic_time[1:]-cosmic_time[:-1]))]
#    
#    if(metallicity_probe=='OH' or metallicity_probe=='FOH'\
#        or metallicity_probe=='12+log(O/H)' or metallicity_probe=='[O/H]'):
#        if(cluster_data):
#            input_file=data_path+str(model_name)+'/'+'FOH_z_dM_clusters_'+cluster_str+'.dat'
#        else: input_file=data_path+str(model_name)+'/'+'FOH_z_dM.dat'
#        OH_arr = FOH_arr - float(FOH_binning[FOH_binning_desc.index('solar ref. value')])
#        XH_arr=OH_arr
#    elif(metallicity_probe=='FeH' or metallicity_probe=='FFeH'\
#        or metallicity_probe=='12+log(Fe/H)' or metallicity_probe=='[Fe/H]'):
#        if(cluster_data):
#            input_file=data_path+str(model_name)+'/'+'FeH_z_dM_clusters_'+cluster_str+'.dat'        
#        else: input_file=data_path+str(model_name)+'/'+'FFeH_z_dM.dat'
#        XH_arr=FeH_arr        
#    elif(metallicity_probe=='OFe' or metallicity_probe=='[O/Fe]'):
#        if(cluster_data):
#            input_file=data_path+str(model_name)+'/'+'OFe_z_dM_clusters_'+cluster_str+'.dat'        
#        else: input_file=data_path+str(model_name)+'/'+'OFe_z_dM.dat'
#        XH_arr=OFe_arr
#    dXH=[*[XH_arr[1]-XH_arr[0]],*(np.abs(XH_arr[1:]-XH_arr[:-1]))]
#    if(cluster_data):
#        cluster_mass_bins=get_cluster_mass_bins(cluster_mass_binning=cluster_mass_binning,\
#            cluster_mass_binning_desc=cluster_mass_binning_desc)
#        bin_data = read_data_for_cluster_mass_bin(file_path=input_file, bin_edges=cluster_mass_bins,\
#                   value=bin_val, dt=dt)
##        bin_data = read_data_for_cluster_mass_bin(file_path=input_file, bin_no=bin_idx, dt=dt)
#        data=np.array(bin_data)
#        #print(data.shape)
#    else:    
#        f=open(input_file,'r')
#        data = np.array( [ [float(data)/(1e6*dti) for data in line.split()]\
#                        for line,dti in zip(f.readlines(),delt) ]
#                        )
#        f.close()
#    if(smmothit): SFRD_data = gaussian_filter(data, sigma=2)
#    else:    SFRD_data=data

#    if(Xieff!='none'):
#            #weight the data by formation efficiency vs metallicity
#            if(Xieff=='Logistic_function'):
#                            filter_metallicity=np.zeros(SFRD_data.shape)
#                            x0=Xeff_params[Xeff_params_desc.index('x0')]
#                            k=Xeff_params[Xeff_params_desc.index('k')]
#                            L=Xeff_params[Xeff_params_desc.index('L')]
#                            C=Xeff_params[Xeff_params_desc.index('C')]                            
#                            for x,ii in zip(XH_arr,range(len(XH_arr))):
#                                filter_metallicity[:,ii] = Logistic_function(x,L,x0,k,C)
#            SFRD_data*=filter_metallicity
#    
#    if(info_print): 
#        print( '(#timesteps,#ZZ bins)=',SFRD_data.shape,\
#             (SFRD_data[:][0]).shape, SFRD_data.shape[1],SFRD_data.shape[0],\
#             'dt table len:',len(delt))
#    idx_upper_cut, arr_val_upper_cut = find_nearest(XH_arr,XH_upper_cut)
#    idx_lower_cut, arr_val_lower_cut = find_nearest(XH_arr,XH_lower_cut)
#    mtot_z=[]
#    for zi in range(len(redshifts)):
#        SFRD_z_in_metallicity_range = np.sum((SFRD_data[zi])[idx_lower_cut:idx_upper_cut+1])
#        mtot_z.append(SFRD_z_in_metallicity_range)
#    
#    if(info_print): print('min val:', XH_arr[0],'max val:', XH_arr[-1],\
#          'min val of cut:', arr_val_lower_cut, 'max val of cut:',arr_val_upper_cut,\
#          idx_lower_cut,idx_upper_cut)
#          
#    SFRD_data_dZ = np.array( [ [float(data)/(dZ) for data,dZ in zip(SFRD_data[:][ii],dXH)]\
#                             for ii in range(len(SFRD_data[:,0])) ] )          

#    XHi_SFRD_weighted = np.array( [ [(float(data)*Z)/np.sum(SFRD_data[:][ii])\
#                             for data,Z in zip(SFRD_data[:][ii],XH_arr)]\
#                             for ii in range(len(SFRD_data[:,0])) ] ) 
#    XH_SFRD_weighted=[]
#    for zi in range(len(redshifts)):
#        XH_SFRD_weighted.append(np.sum(XHi_SFRD_weighted[zi]))
#          
#    lbt=max(cosmic_time)-cosmic_time
#    return SFRD_data_dZ, mtot_z, redshifts[::-1],lbt[::-1], cosmic_time[::-1], delt, XH_arr, XH_SFRD_weighted

def compute_percentile_XH_(SFRD_data, XH_arr, redshifts, percentile=0.5):
    percentile_XH = []
    for zi in range(len(redshifts)):
        sfrd = np.array(SFRD_data[zi])
        xh = np.array(XH_arr)

        # Sort metallicity and SFRD accordingly
        sort_idx = np.argsort(xh)
        sfrd_sorted = sfrd[sort_idx]
        xh_sorted = xh[sort_idx]

        # Cumulative distribution of SFRD
        sfrd_cumsum = np.cumsum(sfrd_sorted)
        sfrd_cumsum /= sfrd_cumsum[-1]  # Normalize to 1

        # Find the XH where cumulative SFRD exceeds the percentile
        idx = np.searchsorted(sfrd_cumsum, percentile)
        percentile_XH.append(xh_sorted[min(idx, len(xh_sorted)-1)])

    return percentile_XH


def compute_percentile_XH(SFRD_data,XH_arr, redshifts, percentile=0.9):
    percentile_XH = []
    for zi in range(len(redshifts)):
        sfrd = np.array(SFRD_data[zi])
        xh = np.array(XH_arr)
        mtot_z = np.sum(sfrd)
        sfrd_normed = sfrd/mtot_z
        sfrd_cumsum=[np.sum(sfrd_normed[:j]) for j in range(xh.shape[0])]
        idx, arr_val = find_nearest(sfrd_cumsum, percentile)
        #print(arr_val, percentile)
        percentile_XH.append(xh[min(idx, len(xh)-1)])
    return percentile_XH
    
def get_data_cut_in_metallicity(metallicity_probe='FeH', XH_lower_cut=-5, XH_upper_cut=5,
                                model_name='', data_path='',\
                                cosmic_time=[], redshifts=[], dt=60., info_print=False, smmoothit=False,
                                Xieff='none', Xeff_params_desc=[], Xeff_params=[],
                                cluster_data=False, cluster_str='form', bin_val=10**5.,
                                cluster_mass_binning=[10**3.5, 1e9, 13],
                                cluster_mass_binning_desc=['min value', 'max value', 'number of bins'],\
                                fraction_of_SB_SF=0.5,data_operation="subtract model2"):
#                                "lower SB contribution"):
    
    # Ensure two model names are provided if model_names is a list
    if isinstance(model_name, list):
        if len(model_name) == 3:
            model_name1, model_name2, model_name3 = model_name
        elif len(model_name) == 2:
            model_name1, model_name2 = model_name
            model_name3 = None
        else:
            raise ValueError("If providing a list for model_name, it must contain 2 or 3 model names.")
    elif isinstance(model_name, str) and model_name:
        model_name1 = model_name
        model_name2 = None
        model_name3 = None        
    else:
        raise ValueError("Either a single model_name as a string or a list of 2-3 model_names must be provided.")

    # Get metallicity binning
    FOH_binning, FeH_binning, OFe_binning = get_metallicity_bins()
    FOH_binning_desc, FeH_binning_desc, OFe_binning_desc = get_metallicity_bins(description=True)
    FOH_arr, FeH_arr, OFe_arr = get_metallicity_bins(arrays=True)
    
    # Calculate time differences
    delt = [*[dt], *(np.abs(cosmic_time[1:] - cosmic_time[:-1]))]
    
    # Determine which metallicity probe to use
    if metallicity_probe in ['OH', 'FOH', '12+log(O/H)', '[O/H]']:
        if cluster_data:
            file_suffix = f"FOH_z_dM_clusters_{cluster_str}.dat"
        else:
            file_suffix = "FOH_z_dM.dat"
        OH_arr = FOH_arr - float(FOH_binning[FOH_binning_desc.index('solar ref. value')])
        XH_arr = OH_arr
    elif metallicity_probe in ['FeH', 'FFeH', '12+log(Fe/H)', '[Fe/H]']:
        if cluster_data:
            file_suffix = f"FeH_z_dM_clusters_{cluster_str}.dat"
        else:
            file_suffix = "FFeH_z_dM.dat"
        XH_arr = FeH_arr
    elif metallicity_probe in ['OFe', '[O/Fe]']:
        if cluster_data:
            file_suffix = f"OFe_z_dM_clusters_{cluster_str}.dat"
        else:
            file_suffix = "OFe_z_dM.dat"
        XH_arr = OFe_arr
    else:
        raise ValueError("Invalid metallicity probe specified.")

    dXH = [*[XH_arr[1] - XH_arr[0]], *(np.abs(XH_arr[1:] - XH_arr[:-1]))]

    # Read data for a single model or two models and compute difference
    def read_model_data(input_file):
        if cluster_data:
            cluster_mass_bins = get_cluster_mass_bins(cluster_mass_binning=cluster_mass_binning,
                                                      cluster_mass_binning_desc=cluster_mass_binning_desc)
            bin_data = read_data_for_cluster_mass_bin(file_path=input_file, bin_edges=cluster_mass_bins,
                                                      value=bin_val, dt=dt)
            return np.array(bin_data)
        else:
            with open(input_file, 'r') as f:
                data = np.array([[float(val) / (1e6 * dti) for val in line.split()]
                                 for line, dti in zip(f.readlines(), delt)])
            return data
    if model_name3:  # If three models are provided, compute the difference
        input_files = [f"{data_path}{model}/{file_suffix}" for model in [model_name1, model_name2,model_name3]]
        data_model1 = read_model_data(input_files[0])
        data_model2 = read_model_data(input_files[1])
        data_model3 = read_model_data(input_files[2])        
        if(data_operation=="subtract model2"): data = data_model1 - data_model2
        elif(data_operation=="subtract model1"): data = data_model2 - data_model1
        elif(data_operation=="fraction_of_SB_SF*(model1-model2)+model3"): 
            data_only_SB = data_model1 - data_model2
            data = data_model3 + fraction_of_SB_SF*data_only_SB
            if info_print:
                print('new fraction of SB contribution: ',fraction_of_SB_SF ,\
                'assum. high SB model= ',model_name1)
        else: data = data_model1
    elif model_name2:  # If two models are provided, compute the difference
        input_files = [f"{data_path}{model}/{file_suffix}" for model in [model_name1, model_name2]]
        data_model1 = read_model_data(input_files[0])
        data_model2 = read_model_data(input_files[1])
        if(data_operation=="subtract model2"): data = data_model1 - data_model2
        elif(data_operation=="subtract model1"): data = data_model2 - data_model1
        elif(data_operation=="lower SB contribution"): 
            data_only_SB = data_model1 - data_model2
            data = data_model2 + fraction_of_SB_SF*data_only_SB
            if info_print:
                print('new fraction of SB contribution: ',fraction_of_SB_SF ,\
                'assum. high SB model= ',model_name1)
        else: data = data_model1
    else:  # If only one model is provided
        input_file = f"{data_path}{model_name1}/{file_suffix}"
        data = read_model_data(input_file)

    # Apply smoothing if specified
    if smmoothit:
        SFRD_data = gaussian_filter(data, sigma=2)
    else:
        SFRD_data = data

    # Apply weighting by formation efficiency if specified
    if Xieff != 'none':
        if Xieff == 'Logistic_function':
            filter_metallicity = np.zeros(SFRD_data.shape)
            x0 = Xeff_params[Xeff_params_desc.index('x0')]
            k = Xeff_params[Xeff_params_desc.index('k')]
            L = Xeff_params[Xeff_params_desc.index('L')]
            C = Xeff_params[Xeff_params_desc.index('C')]
            if('X_truncate' in Xeff_params_desc): 
                X_truncate = Xeff_params[Xeff_params_desc.index('X_truncate')]
                for x, ii in zip(XH_arr, range(len(XH_arr))):
                    if(x>X_truncate): filter_metallicity[:, ii]=0
                    else: filter_metallicity[:, ii] = Logistic_function(x, L, x0, k, C)
            else:
                for x, ii in zip(XH_arr, range(len(XH_arr))):
                    filter_metallicity[:, ii] = Logistic_function(x, L, x0, k, C)                    
            SFRD_data *= filter_metallicity

    if info_print:
        print('(Timesteps, ZZ bins) =', SFRD_data.shape, 'dt table length:', len(delt))

    # Find indices for the metallicity cuts
    idx_upper_cut, arr_val_upper_cut = find_nearest(XH_arr, XH_upper_cut)
    idx_lower_cut, arr_val_lower_cut = find_nearest(XH_arr, XH_lower_cut)

    #print(SFRD_data)
    # Calculate SFRD within the metallicity range
    mtot_z = []
    for zi in range(len(redshifts)):
        SFRD_z_in_metallicity_range = np.sum(SFRD_data[zi][idx_lower_cut:idx_upper_cut + 1])
        mtot_z.append(SFRD_z_in_metallicity_range)

    if info_print:
        print('Metallicity cuts:',
              'min val:', XH_arr[0], 'max val:', XH_arr[-1],
              'cut range:', arr_val_lower_cut, arr_val_upper_cut)

    # Calculate SFRD weighted by dZ
    SFRD_data_dZ = np.array([[float(data) / dZ for data, dZ in zip(SFRD_data[ii], dXH)]
                             for ii in range(len(SFRD_data[:, 0]))])

    # Calculate XH-weighted SFRD
    XHi_SFRD_weighted = np.array([[float(data) * Z / np.sum(SFRD_data[ii])
                                   for data, Z in zip(SFRD_data[ii], XH_arr)]
                                  for ii in range(len(SFRD_data[:, 0]))])
    XH_SFRD_weighted = [np.sum(XHi_SFRD_weighted[zi]) for zi in range(len(redshifts))]

    # Convert cosmic time to look-back time
    lbt = max(cosmic_time) - cosmic_time

    return SFRD_data_dZ, mtot_z, redshifts[::-1], lbt[::-1], cosmic_time[::-1], delt, XH_arr, XH_SFRD_weighted

def get_data_cut_in_redshift(metallicity_probe='FeH', z_lower_cut=0, z_upper_cut=1,
                                model_name='', data_path='',\
                                cosmic_time=[], redshifts=[], dt=60., info_print=False, smmoothit=False,
                                Xieff='none', Xeff_params_desc=[], Xeff_params=[],
                                cluster_data=False, cluster_str='form', bin_val=10**5.,
                                cluster_mass_binning=[10**3.5, 1e9, 13],
                                cluster_mass_binning_desc=['min value', 'max value', 'number of bins'],\
                                fraction_of_SB_SF=0.5,data_operation="subtract model2"):
#                                "lower SB contribution"):
    
    # Ensure two model names are provided if model_names is a list
    if isinstance(model_name, list):
        if len(model_name) == 3:
            model_name1, model_name2, model_name3 = model_name
        elif len(model_name) == 2:
            model_name1, model_name2 = model_name
            model_name3 = None
        else:
            raise ValueError("If providing a list for model_name, it must contain 2 or 3 model names.")
    elif isinstance(model_name, str) and model_name:
        model_name1 = model_name
        model_name2 = None
        model_name3 = None        
    else:
        raise ValueError("Either a single model_name as a string or a list of 2-3 model_names must be provided.")

    # Get metallicity binning
    FOH_binning, FeH_binning, OFe_binning = get_metallicity_bins()
    FOH_binning_desc, FeH_binning_desc, OFe_binning_desc = get_metallicity_bins(description=True)
    FOH_arr, FeH_arr, OFe_arr = get_metallicity_bins(arrays=True)
    
    # Calculate time differences
    delt = [*[dt], *(np.abs(cosmic_time[1:] - cosmic_time[:-1]))]
    
    # Determine which metallicity probe to use
    if metallicity_probe in ['OH', 'FOH', '12+log(O/H)', '[O/H]']:
        if cluster_data:
            file_suffix = f"FOH_z_dM_clusters_{cluster_str}.dat"
        else:
            file_suffix = "FOH_z_dM.dat"
        OH_arr = FOH_arr - float(FOH_binning[FOH_binning_desc.index('solar ref. value')])
        XH_arr = OH_arr
    elif metallicity_probe in ['FeH', 'FFeH', '12+log(Fe/H)', '[Fe/H]']:
        if cluster_data:
            file_suffix = f"FFeH_z_dM_clusters_{cluster_str}.dat"
        else:
            file_suffix = "FFeH_z_dM.dat"
        XH_arr = FeH_arr
    elif metallicity_probe in ['OFe', '[O/Fe]']:
        if cluster_data:
            file_suffix = f"OFe_z_dM_clusters_{cluster_str}.dat"
        else:
            file_suffix = "OFe_z_dM.dat"
        XH_arr = OFe_arr
    else:
        raise ValueError("Invalid metallicity probe specified.")

    dXH = [*[XH_arr[1] - XH_arr[0]], *(np.abs(XH_arr[1:] - XH_arr[:-1]))]

    # Read data for a single model or two models and compute difference
    def read_model_data(input_file):
        if cluster_data:
            cluster_mass_bins = get_cluster_mass_bins(cluster_mass_binning=cluster_mass_binning,
                                                      cluster_mass_binning_desc=cluster_mass_binning_desc)
            bin_data = read_data_for_cluster_mass_bin(file_path=input_file, bin_edges=cluster_mass_bins,
                                                      value=bin_val, dt=dt)
            return np.array(bin_data)
        else:
            with open(input_file, 'r') as f:
                data = np.array([[float(val) / (1e6 * dti) for val in line.split()]
                                 for line, dti in zip(f.readlines(), delt)])
            return data
    if model_name3:  # If three models are provided, compute the difference
        input_files = [f"{data_path}{model}/{file_suffix}" for model in [model_name1, model_name2,model_name3]]
        data_model1 = read_model_data(input_files[0])
        data_model2 = read_model_data(input_files[1])
        data_model3 = read_model_data(input_files[2])        
        if(data_operation=="subtract model2"): data = data_model1 - data_model2
        elif(data_operation=="subtract model1"): data = data_model2 - data_model1
        elif(data_operation=="fraction_of_SB_SF*(model1-model2)+model3"): 
            data_only_SB = data_model1 - data_model2
            data = data_model3 + fraction_of_SB_SF*data_only_SB
            if info_print:
                print('new fraction of SB contribution: ',fraction_of_SB_SF ,\
                'assum. high SB model= ',model_name1)
        else: data = data_model1
    elif model_name2:  # If two models are provided, compute the difference
        input_files = [f"{data_path}{model}/{file_suffix}" for model in [model_name1, model_name2]]
        data_model1 = read_model_data(input_files[0])
        data_model2 = read_model_data(input_files[1])
        if(data_operation=="subtract model2"): data = data_model1 - data_model2
        elif(data_operation=="subtract model1"): data = data_model2 - data_model1
        elif(data_operation=="lower SB contribution"): 
            data_only_SB = data_model1 - data_model2
            data = data_model2 + fraction_of_SB_SF*data_only_SB
            if info_print:
                print('new fraction of SB contribution: ',fraction_of_SB_SF ,\
                'assum. high SB model= ',model_name1)
        else: data = data_model1
    else:  # If only one model is provided
        input_file = f"{data_path}{model_name1}/{file_suffix}"
        data = read_model_data(input_file)

    # Apply smoothing if specified
    if smmoothit:
        SFRD_data = gaussian_filter(data, sigma=2)
    else:
        SFRD_data = data

    # Apply weighting by formation efficiency if specified
    if Xieff != 'none':
        if Xieff == 'Logistic_function':
            filter_metallicity = np.zeros(SFRD_data.shape)
            x0 = Xeff_params[Xeff_params_desc.index('x0')]
            k = Xeff_params[Xeff_params_desc.index('k')]
            L = Xeff_params[Xeff_params_desc.index('L')]
            C = Xeff_params[Xeff_params_desc.index('C')]
            if('X_truncate' in Xeff_params_desc): 
                X_truncate = Xeff_params[Xeff_params_desc.index('X_truncate')]
                for x, ii in zip(XH_arr, range(len(XH_arr))):
                    if(x>X_truncate): filter_metallicity[:, ii]=0
                    else: filter_metallicity[:, ii] = Logistic_function(x, L, x0, k, C)
            else:
                for x, ii in zip(XH_arr, range(len(XH_arr))):
                    filter_metallicity[:, ii] = Logistic_function(x, L, x0, k, C)                    
            SFRD_data *= filter_metallicity

    if info_print:
        print('(Timesteps, ZZ bins) =', SFRD_data.shape, 'dt table length:', len(delt))

    # Find indices for the redshift cuts
    idx_upper_cut, arr_val_upper_cut = find_nearest(redshifts[::-1], z_upper_cut)
    idx_lower_cut, arr_val_lower_cut = find_nearest(redshifts[::-1], z_lower_cut)
    if(info_print): print(redshifts[::-1][idx_upper_cut], redshifts[::-1][idx_lower_cut])
    z_idx_range=np.arange(start=min(idx_lower_cut,idx_upper_cut),\
                        stop=max(idx_lower_cut,idx_upper_cut)+1, step=1)
    # Calculate SFRD weighted by dZ
    data_z_range=np.zeros(XH_arr.shape)
    for ii in z_idx_range:
        data_z_range+=SFRD_data[ii]/dXH
#        data_z_range+=SFRD_data[len(SFRD_data[:, 0])-ii-1]/dXH 

    # Convert cosmic time to look-back time
    lbt = max(cosmic_time) - cosmic_time
    return SFRD_data, redshifts[::-1], lbt[::-1], cosmic_time[::-1], delt, XH_arr, data_z_range

def load_cluster_evol_data(metallicity_probe='FeH', XH_lower_cut=-5, XH_upper_cut=5,
                       model_name='', data_path='',\
                       cosmic_time=[], redshifts=[], dt=60.,\
                       cluster_data=True, cluster_str='evol',\
                       cluster_mass_binning=[10**3.5, 1e9, 13],\
                       cluster_mass_binning_desc=['min value', 'max value', 'number of bins'],\
                       selected_mass_bins=None,  timestep_range=None, info=True,smmoothit=False):

    XH_range=(XH_lower_cut,XH_upper_cut)
    total_timesteps=len(redshifts)
    
    if cluster_data:
            cluster_mass_bins = get_cluster_mass_bins(cluster_mass_binning=cluster_mass_binning,
                                                      cluster_mass_binning_desc=cluster_mass_binning_desc)
    # Get metallicity binning
    FOH_binning, FeH_binning, OFe_binning = get_metallicity_bins()
    FOH_binning_desc, FeH_binning_desc, OFe_binning_desc = get_metallicity_bins(description=True)
    FOH_arr, FeH_arr, OFe_arr = get_metallicity_bins(arrays=True)
    
    # Calculate time differences
    delt = [*[dt], *(np.abs(cosmic_time[1:] - cosmic_time[:-1]))]
    
    # Determine which metallicity probe to use
    if metallicity_probe in ['OH', 'FOH', '12+log(O/H)', '[O/H]']:
        if cluster_data:
            file_suffix = f"FOH_z_dM_clusters_{cluster_str}.dat"
        else:
            file_suffix = "FOH_z_dM.dat"
        OH_arr = FOH_arr - float(FOH_binning[FOH_binning_desc.index('solar ref. value')])
        XH_arr = OH_arr
        XH_bin_edges = np.linspace(min(XH_arr), max(XH_arr),len(XH_arr)+1)        
    elif metallicity_probe in ['FeH', 'FFeH', '12+log(Fe/H)', '[Fe/H]']:
        if cluster_data:
            file_suffix = f"FeH_z_dM_clusters_{cluster_str}.dat"
        else:
            file_suffix = "FFeH_z_dM.dat"
        XH_arr = FeH_arr
        XH_bin_edges = np.linspace(min(XH_arr), max(XH_arr),len(XH_arr)+1)
    elif metallicity_probe in ['OFe', '[O/Fe]']:
        if cluster_data:
            file_suffix = f"OFe_z_dM_clusters_{cluster_str}.dat"
        else:
            file_suffix = "OFe_z_dM.dat"
        XH_arr = OFe_arr
    else:
        raise ValueError("Invalid metallicity probe specified.")
                               
    n_foh = len(XH_arr)
    n_m = len(cluster_mass_bins) - 1
    n_t = total_timesteps
    input_file=data_path+model_name+'/'+file_suffix

    # Load data
    data = np.loadtxt(input_file)    
    #mass density **per unit time [yr]**
    data/=(1e6 * dt)

    if(info):    print(data.shape)
    assert data.shape == (n_t, n_foh * n_m), "Mismatch in data shape"
    # Reshape into (foh_idx, m_idx, t)
    data = data.T.reshape(n_foh, n_m, n_t)


    # Apply smoothing if specified
    if smmoothit:
        SFRD_data = gaussian_filter(data, sigma=2)
    else:
        SFRD_data = data

    dXH = [*[XH_arr[1] - XH_arr[0]], *(np.abs(XH_arr[1:] - XH_arr[:-1]))]
    d_XH= np.array([dXHi for dXHi in dXH])[:, np.newaxis, np.newaxis]
    SFRD_data_dZ = SFRD_data / d_XH


    # --- Apply filters ---
    # Filter mass bins
    if selected_mass_bins is not None:
        SFRD_data_dZ = SFRD_data_dZ[:, selected_mass_bins, :]
        SFRD_data = SFRD_data[:, selected_mass_bins, :]
        mass_bin_labels = [f"{cluster_mass_bins[i]:.2e}-{cluster_mass_bins[i+1]:.2e}" 
                           for i in selected_mass_bins]
    else:
        selected_mass_bins = list(range(n_m))
        mass_bin_labels = [f"{cluster_mass_bins[i]:.2e}-{cluster_mass_bins[i+1]:.2e}" 
                           for i in range(n_m)]

    # Filter metallicity bins
    if XH_range is not None:
        XH_min, XH_max = XH_range
        selected_XH_indices = [i for i in range(n_foh)
                                if XH_bin_edges[i] >= XH_min and XH_bin_edges[i+1] <= XH_max]
        SFRD_data_dZ = SFRD_data_dZ[selected_XH_indices, :, :]
        SFRD_data=SFRD_data[selected_XH_indices, :, :]       
        XH_bin_labels = [f"{XH_bin_edges[i]:.2f} to {XH_bin_edges[i+1]:.2f}" 
                          for i in selected_XH_indices]
    else:
        XH_bin_labels = [f"{XH_bin_edges[i]:.2f} to {XH_bin_edges[i+1]:.2f}" 
                          for i in range(n_foh)]
    # Filter timesteps
    if timestep_range is not None:
        t_min, t_max = timestep_range
        SFRD_data_dZ = SFRD_data_dZ[:, :, t_min:t_max+1]
        SFRD_data=SFRD_data[:, :, t_min:t_max+1]
    else:
        t_min, t_max = 0, n_t - 1

    mtot_z = np.sum(SFRD_data, axis=(0, 1))  # shape: (filtered_timesteps,)

    # Reshape for broadcasting: (n_FeH_filtered, 1, 1)    
    Z=XH_arr[selected_XH_indices, np.newaxis, np.newaxis]
    # Calculate SFRD weighted by dZ
    # Compute numerator: sum over (mass * XH_center)
    weighted_XH_sum = np.sum(SFRD_data * Z, axis=(0, 1))  # shape: (n_timesteps,)
    # Compute mass-weighted FeH: [Fe/H]_mean(t)
    XH_SFRD_weighted = weighted_XH_sum / mtot_z  # shape: (n_timesteps,)
#        
    # Convert cosmic time to look-back time
    lbt = max(cosmic_time) - cosmic_time
    
    return {
        "data": SFRD_data_dZ,  # Shape: (filtered_feh, filtered_mass, filtered_timesteps)
        "FeH_bins": XH_bin_labels,
        "mass_bins": mass_bin_labels,
        "timesteps": list(range(t_min, t_max + 1)),
        "SFH": mtot_z,  # 1D array: total mass density formed at time/redshift
        "SFRD_weighted_mean_XH": XH_SFRD_weighted,
        "lookback time": lbt[::-1],
        "XH_arr": XH_arr,
        "redshifts": redshifts[::-1],
        "cosmic time": cosmic_time[::-1],
    }
    

def select_cluster_mass_bins(mass_cut,sign=None,\
        cluster_mass_binning=[10**3.5, 1e9, 13],\
        cluster_mass_binning_desc=['min value', 'max value', 'number of bins']):

    cluster_mass_bins = get_cluster_mass_bins(cluster_mass_binning=cluster_mass_binning,
                        cluster_mass_binning_desc=cluster_mass_binning_desc)
    bin_centers=(cluster_mass_bins[1:]+cluster_mass_bins[:-1])*0.5
    flag=None
    
    if(len(mass_cut)==2): 
        bin_idx=find_bin_index(bin_edges=cluster_mass_bins, value=mass_cut[0])
        bin_idx2=find_bin_index(bin_edges=cluster_mass_bins, value=mass_cut[1])
        if bin_idx == -1 or bin_idx2==-1:
            print('value out of range')
            exit()
        #bin_vals=bin_centers[bin_idx:bin_idx2]
        #bin_indices=np.arange(bin_idx,bin_idx2)
        if(bin_idx==bin_idx2):
                bin_vals=bin_centers[bin_idx:bin_idx+1]
                bin_indices=[bin_idx]
                str_cluster_m_range = r'log$_{10}$(M$_{\rm cl}$/M$_{\odot}$)= '+\
                str('%.2g'%np.log10(cluster_mass_bins[bin_idx]))+\
                ' - '+str('%.2g'%np.log10(cluster_mass_bins[bin_idx+1]))   
        else:    
            bin_vals=bin_centers[bin_idx:bin_idx2]
            bin_indices=np.arange(bin_idx,bin_idx2)
            str_cluster_m_range = r'log$_{10}$(M$_{\rm cl}$/M$_{\odot}$)= '+\
                str('%.2g'%np.log10(cluster_mass_bins[bin_idx]))+\
                ' - '+str('%.2g'%np.log10(cluster_mass_bins[bin_idx2]))        
        print(str_cluster_m_range)                
    else:    
        bin_idx=find_bin_index(bin_edges=cluster_mass_bins, value=mass_cut)
        if(sign=='>' or sign=='geq'):
            bin_vals=bin_centers[bin_idx:]
            bin_indices=np.arange(bin_idx, len(cluster_mass_bins)-1)            
            str_cluster_m_range = r'log$_{10}$(M$_{\rm cl}$/M$_{\odot}$)> '+\
            str('%.2g'%np.log10(cluster_mass_bins[bin_idx]))
            print(str_cluster_m_range)            
        elif(sign=='<' or sign=='leq'):
            if(mass_cut[0]>10**lMmin_cluster):
                bin_vals=bin_centers[:bin_idx]
                bin_indices=np.arange(0,bin_idx)                
                str_cluster_m_range = r'log$_{10}$(M$_{\rm cl}$/M$_{\odot}$)< '+\
                str('%.2g'%np.log10(cluster_mass_bins[bin_idx]))          
                print(str_cluster_m_range)
            else: 
                print("\n (!) the selected mass cut is below minimum mass lMmin_cluster=", lMmin_cluster )
                print(" all the remaining 'field' SFRD([X/H], z) will be plotted (!) ")
                str_cluster_m_range = r'log$_{10}$(M$_{\rm cluster}$/M$_{\odot}$)< '+\
                str('%.2g'%lMmin_cluster)
                print(str_cluster_m_range ,'\n')

                sign='>'
                mass_cut=[10**lMmin_cluster]
                bin_idx=find_bin_index(bin_edges=cluster_mass_bins, value=mass_cut)
                bin_vals=bin_centers[bin_idx:]
                bin_indices=[]
                flag='plot_SFRD_dissolved_to_field'                            
        else:
            bin_vals=[mass_cut]
            bin_indices=find_bin_index(bin_edges=cluster_mass_bins, value=mass_cut)
            str_cluster_m_range = r'log$_{10}$(M$_{\rm cl}$/M$_{\odot}$)= '+\
                str('%.2g'%np.log10(cluster_mass_bins[bin_idx]))+\
                ' - '+str('%.2g'%np.log10(cluster_mass_bins[bin_idx+1]))
            print(str_cluster_m_range) 
    return bin_indices, bin_vals, flag, str_cluster_m_range
   
def stack_cluster_mass_bins(model_name, mass_cut=[10**5],sign='>',\
    metallicity_probe='FeH',XH_lower_cut=-5.,XH_upper_cut=5.,\
            data_path='../data',\
            cosmic_time=[],redshifts=[],\
    cluster_str='form',bin_val=10**5.,\
    cluster_mass_binning=[10**3.5,1e9,13],\
    cluster_mass_binning_desc=['min value', 'max value','number of bins']):
    
    ''' Find bins with interesting cluster masses '''
    cluster_mass_bins = get_cluster_mass_bins(cluster_mass_binning=cluster_mass_binning,\
        cluster_mass_binning_desc=cluster_mass_binning_desc)
    bin_centers=(cluster_mass_bins[1:]+cluster_mass_bins[:-1])*0.5

    if(len(mass_cut)==2): 
        bin_idx=find_bin_index(bin_edges=cluster_mass_bins, value=min(mass_cut))
        bin_idx2=find_bin_index(bin_edges=cluster_mass_bins, value=max(mass_cut))
        if bin_idx == -1 or bin_idx2==-1:
            print('value out of range')
            exit()        
        if(bin_idx2==bin_idx): bin_idx2=bin_idx+1
        bin_vals=bin_centers[bin_idx:bin_idx2]
        str_cluster_m_range = r'log$_{10}$(M$_{\rm cl}$/M$_{\odot}$)= '+\
            str('%.2g'%np.log10(cluster_mass_bins[bin_idx]))+\
            ' - '+str('%.2g'%np.log10(cluster_mass_bins[bin_idx2]))
        #print(str_cluster_m_range, bin_vals)
    else:    
        bin_idx=find_bin_index(bin_edges=cluster_mass_bins, value=mass_cut)
        if(sign=='>' or sign=='geq'):
            bin_vals=bin_centers[bin_idx:]
            str_cluster_m_range = r'log$_{10}$(M$_{\rm cl}$/M$_{\odot}$)> '+\
            str('%.2g'%np.log10(cluster_mass_bins[bin_idx]))
            print(str_cluster_m_range)            
        elif(sign=='<' or sign=='leq'):
            bin_vals=bin_centers[:bin_idx]
            str_cluster_m_range = r'log$_{10}$(M$_{\rm cl}$/M$_{\odot}$)< '+\
            str('%.2g'%np.log10(cluster_mass_bins[bin_idx]))          
            print(str_cluster_m_range)                  
        else:
            bin_vals=[mass_cut]
            str_cluster_m_range = r'log$_{10}$(M$_{\rm cl}$/M$_{\odot}$)= '+\
                str('%.2g'%np.log10(cluster_mass_bins[bin_idx]))+\
                ' - '+str('%.2g'%np.log10(cluster_mass_bins[bin_idx+1]))
            print(str_cluster_m_range)    

    i=0
    for bin_val in bin_vals:

        SFRD_data_bin, SFRD_z_tot_bin,zz,lbt,c_time,delt, XH_arr,XH_SFRD_weighted=\
                get_data_cut_in_metallicity(metallicity_probe=metallicity_probe,\
                XH_lower_cut=XH_lower_cut,XH_upper_cut=XH_upper_cut,\
                model_name=model_name,\
                data_path=data_path,\
                cosmic_time=cosmic_time,redshifts=redshifts,\
                cluster_data=True,cluster_str=cluster_str,\
                bin_val=bin_val,\
                cluster_mass_binning=cluster_mass_binning,\
                cluster_mass_binning_desc=cluster_mass_binning_desc)
                
        if(i==0): 
            SFRD_data=SFRD_data_bin
            SFRD_z_tot_=np.array(SFRD_z_tot_bin)
        else: 
            SFRD_data+=SFRD_data_bin
            SFRD_z_tot_+=np.array(SFRD_z_tot_bin)
        i+=1
    return SFRD_data,SFRD_z_tot_,lbt,c_time,delt, XH_arr, str_cluster_m_range    
    
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
    

def extract_model_parameters(model_name):
    #    # Example usage:
    #    model_name = get_model_name(SFMR_ref='B18')
    #    parameters = extract_model_parameters(model_name)
    #    SFMR_ref=parameters['SFMR_ref']
    #    print(model_name, SFMR_ref) 

    # Define example values for parameters
    examples = {
        'SFMR_ref': 'P23',
        'calib': 'C20_ADF',
        'MZR_evol': 'False',
        'delta_FMR_ZOH_asym_0': 0.,
        'evolving_low_M_slope': 'True',
        'OFe_ref': 'pl400CCFep1',
        'FMR_slope': 0.27,
        'SB_ref': 'Boco',
        'lMmin' : 6.,
        'lMmax' : 12.,
        'z_start': 10.,
        'IMF_ref': 'K01',
        'cluster_ref' : 'None',
        'any_addons': 'None'
    }

    # Initialize the parameters dictionary with example values
    params = examples.copy()
    # Split the model_name string by '-'
    parts = model_name.split('-')

    # Extract the parameters from the parts
    for i in range(0, len(parts), 2):
        if parts[i] == 'SFR':
            params['SFMR_ref'] = parts[i+1]
        elif parts[i] == 'Z':
            params['calib'] = parts[i+1]
        elif parts[i] == 'Zev':
            params['MZR_evol'] = parts[i+1]
        elif parts[i] == 'GSMFev':
            params['evolving_low_M_slope'] = parts[i+1]
        elif parts[i] == 'OFe':
            params['OFe_ref'] = parts[i+1]
        elif parts[i] == 'FMR0':
            params['FMR_slope'] = float('0.' + parts[i+1])
        elif parts[i] == 'dFMR3_10':
            params['delta_FMR_ZOH_asym_0'] = float('0.' + parts[i+1])
        elif parts[i] == 'SB':
            params['SB_ref'] = parts[i+1]
        elif parts[i] == 'IMF':
            params['IMF_ref'] = parts[i+1]
        elif parts[i] == 'add':
            params['any_addons'] = parts[i+1]
        elif parts[i] == 'lMmin':
            params['lMmin'] = float(parts[i+1])
        elif parts[i] == 'lMmax' :
            params['lMmax'] = float(parts[i+1])
        elif parts[i] == 'z_start' :
            params['z_start'] = float(parts[i+1])
        elif parts[i] == 'clusters' :
            params['cluster_ref'] = parts[i+1]

    # Convert string 'True' and 'False' to boolean
    params['MZR_evol'] = (params['MZR_evol'] == 'True')
    params['evolving_low_M_slope'] = (params['evolving_low_M_slope'] == 'True')

    return params

#def get_input_models(SFMR_ref_list=['P23slope08','P23'],\
#     calib_list=['C20_ADF'], MZR_evol_list=[True],\
#     evolving_low_M_slope_list=[True,False],\
#     OFe_ref_list=['pl400CCFep1'], FMR_slope_list=[0.27],\
#     SB_ref_list=['Boco'], IMF_ref_list=['K01'],\
#     lMmin_list=[6.], lMmax_list=[12.], z_start_list=[10.],\
#     any_addons_list=['None'],\
#     full_length=223,\
#     data_path = '/home/martich/Documents/work/properties_of_galaxies_v4/data/'):

#    # List of filenames to check
#    queried_model_names = []
#    # List of files that exist
#    existing_models = []
#    for SFMR_ref in SFMR_ref_list:
#        for calib in calib_list:
#            for MZR_evol in MZR_evol_list:
#                for evolving_low_M_slope in evolving_low_M_slope_list:
#                    for OFe_ref in OFe_ref_list:
#                        for FMR_slope in FMR_slope_list:
#                            for SB_ref in SB_ref_list:
#                                for IMF_ref in IMF_ref_list:
#                                    for lMmin in lMmin_list:
#                                        for lMmax in lMmax_list:
#                                            for z_start in z_start_list:
#                                                for any_addons in any_addons_list:
#                                                    fname = get_model_name(SFMR_ref=SFMR_ref,\
#                                                    calib=calib,MZR_evol=MZR_evol,\
#                                                    evolving_low_M_slope=evolving_low_M_slope,\
#                                                    OFe_ref=OFe_ref,FMR_slope=FMR_slope,\
#                                                    SB_ref=SB_ref,IMF_ref=IMF_ref,\
#                                                    lMmin=lMmin,lMmax=lMmax,z_start=z_start,\
#                                                    any_addons=any_addons)
#                                                    queried_model_names.append(str(fname))
#                                                    input_file =os.path.join(data_path,\
#                                                                 str(fname))
#                                                    if os.path.exists(input_file):
#                                                        with open(input_file+'/FOH_z_dM.dat', 'r') as f:
#                                                            line_count = sum(1 for line in f)
#                                                        if line_count == full_length:
#                                                                existing_models.append(fname)
#                                                        if(fname!=existing_models[-1]):
#                                                            print('running? ',fname)
#    return existing_models, queried_model_names
def get_model_selection(OFe_var='all', OFe_color=['#e66101','#5e3c99','#01665e'],\
    no_GSMF_fix=False, no_asfr_1=False, no_SB=False, skip_excluded_by_lGRB=True,\
    no_delta_FMR_ZOH_asym_0=False):

    model_list=[]
    model_OFe_color=[]
    SB_flag=[]
    lowSB,high_SB='Boco','Rinaldi24'
    calib='C20_ADF'

    if(OFe_var=='long_DTD' or OFe_var=='slow'):
        OFe_ref='pl400CCFep1'
        OFe_ref_list=[OFe_ref]
        OFe_color_list=[OFe_color[1]]
    elif(OFe_var=='short_DTD' or OFe_var=='fast'):
        OFe_ref='pl40CCFep03'
        OFe_ref_list=[OFe_ref]       
        OFe_color_list=[OFe_color[0]]         
    elif(OFe_var=='mixed' or OFe_var=='G05CCFep07'):
        OFe_ref='G05CCFep07'
        OFe_ref_list=[OFe_ref]     
        OFe_color_list=[OFe_color[2]]           
    else:
        OFe_ref_list=['pl40CCFep03','pl400CCFep1','G05CCFep07']
        OFe_color_list=OFe_color
            
    for OFe_ref,OFe_color in zip(OFe_ref_list,OFe_color_list):

        model_list.append( 
                    get_model_name(SFMR_ref='P23slope08', calib=calib,\
                     evolving_low_M_slope=True, OFe_ref=OFe_ref,\
                     FMR_slope=0.27, delta_FMR_ZOH_asym_0=0.25, SB_ref=lowSB)
                     )
        SB_flag.append(0)
        model_OFe_color.append(OFe_color)

        model_list.append( 
                    get_model_name(SFMR_ref='P23_modif_highz', calib=calib,\
                     evolving_low_M_slope=True, OFe_ref=OFe_ref,\
                     FMR_slope=0.27, delta_FMR_ZOH_asym_0=0.25, SB_ref=lowSB)
                     )
        SB_flag.append(0)
        model_OFe_color.append(OFe_color)
        
        if(no_GSMF_fix==False):                     
            model_list.append( 
                        get_model_name(SFMR_ref='P23slope08', calib=calib,\
                         evolving_low_M_slope=False, OFe_ref=OFe_ref,\
                         FMR_slope=0.27, delta_FMR_ZOH_asym_0=0.25, SB_ref=lowSB)
                         )
            model_OFe_color.append(OFe_color)
            SB_flag.append(0)
            
            if(no_asfr_1==False and skip_excluded_by_lGRB==False):
                model_list.append( 
                                get_model_name(SFMR_ref='P23', calib=calib,\
                                 evolving_low_M_slope=False, OFe_ref=OFe_ref,\
                                 FMR_slope=0.27, delta_FMR_ZOH_asym_0=0.25, SB_ref=lowSB)
                                 )
                model_OFe_color.append(OFe_color)
                SB_flag.append(0)
                
        if(no_delta_FMR_ZOH_asym_0==False):                     
            model_list.append( 
                        get_model_name(SFMR_ref='P23slope08', calib=calib,\
                         evolving_low_M_slope=True, OFe_ref=OFe_ref,\
                         FMR_slope=0.27, delta_FMR_ZOH_asym_0=0., SB_ref=lowSB)
                         )                 
            model_OFe_color.append(OFe_color)
            SB_flag.append(0)

        if(no_asfr_1==False):         
            model_list.append( 
                        get_model_name(SFMR_ref='P23', calib=calib,\
                         evolving_low_M_slope=True, OFe_ref=OFe_ref,\
                         FMR_slope=0.27, delta_FMR_ZOH_asym_0=0.25, SB_ref=lowSB)
                         )
            model_OFe_color.append(OFe_color)
            SB_flag.append(0)

            if(no_delta_FMR_ZOH_asym_0==False):
                model_list.append( 
                            get_model_name(SFMR_ref='P23', calib=calib,\
                             evolving_low_M_slope=True, OFe_ref=OFe_ref,\
                             FMR_slope=0.27, delta_FMR_ZOH_asym_0=0., SB_ref=lowSB)
                             )
                model_OFe_color.append(OFe_color)
                SB_flag.append(0)
                        
            if( no_SB==False ):
                model_list.append( 
                                get_model_name(SFMR_ref='P23', calib=calib,\
                                 evolving_low_M_slope=True, OFe_ref=OFe_ref,\
                                 FMR_slope=0.27, delta_FMR_ZOH_asym_0=0.25, SB_ref=high_SB)
                            ) 
                model_OFe_color.append(OFe_color)
                SB_flag.append(1)
                                  
    return model_list, model_OFe_color,SB_flag
    
def get_input_models(SFMR_ref_list=['P23slope08','P23'],\
     calib_list=['C20_ADF'], delta_FMR_ZOH_asym_0_list=[0,0.25],\
     evolving_low_M_slope_list=[True,False],\
     OFe_ref_list=['pl400CCFep1'], FMR_slope_list=[0.27],\
     SB_ref_list=['Boco'], IMF_ref_list=['K01'],\
     lMmin_list=[6.], lMmax_list=[12.], z_start_list=[10.],\
     any_addons_list=['None'],cluster_ref=['None'],\
     full_length=223,\
     data_path = '/home/martich/Documents/work/properties_of_galaxies_v4/data/'):

    # List of filenames to check
    queried_model_names = []
    # List of files that exist
    existing_models = []
    for SFMR_ref in SFMR_ref_list:
        for calib in calib_list:
            for delta_FMR_ZOH_asym_0 in delta_FMR_ZOH_asym_0_list:
                for evolving_low_M_slope in evolving_low_M_slope_list:
                    for OFe_ref in OFe_ref_list:
                        for FMR_slope in FMR_slope_list:
                            for SB_ref in SB_ref_list:
                                for IMF_ref in IMF_ref_list:
                                    for lMmin in lMmin_list:
                                        for lMmax in lMmax_list:
                                            for z_start in z_start_list:
                                                for any_addons in any_addons_list:
                                                    fname = get_model_name(SFMR_ref=SFMR_ref,\
                                                    calib=calib,delta_FMR_ZOH_asym_0=delta_FMR_ZOH_asym_0,\
                                                    evolving_low_M_slope=evolving_low_M_slope,\
                                                    OFe_ref=OFe_ref,FMR_slope=FMR_slope,\
                                                    SB_ref=SB_ref,IMF_ref=IMF_ref,\
                                                    lMmin=lMmin,lMmax=lMmax,z_start=z_start,\
                                                    any_addons=any_addons)
                                                    queried_model_names.append(str(fname))
                                                    input_file =os.path.join(data_path,\
                                                                 str(fname))
                                                    if os.path.exists(input_file):
                                                        with open(input_file+'/FOH_z_dM.dat', 'r') as f:
                                                            line_count = sum(1 for line in f)
                                                        if line_count == full_length:
                                                                existing_models.append(fname)
                                                        if(fname!=existing_models[-1]):
                                                            print('running? ',fname)
    return existing_models, queried_model_names    
