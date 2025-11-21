# from dotenv import load_dotenv
#Make sure to create .env in your 'main_dir' including path to your main_dir
#   e.g. MY_PROJECT_PATH='/default/path/to/folder/'
#then keep the structure of the folders the same as on github and everything should work
import os
# load_dotenv()
#print("MY_PROJECT_PATH:", os.getenv('MY_PROJECT_PATH')) 
# main_dir should be set only from the environment variable
main_dir = os.getenv('MY_PROJECT_PATH', 'MetalCosfr_cp_for_STM_analysis-main/')
tools_dir_path=main_dir+'analysis'
#include path to where the output files for the different model variations are stored
data_path=main_dir+'data/'
import os
import sys
sys.path.append(tools_dir_path)
from MetalCosfr_data_processing_functions import *
from scipy.ndimage import gaussian_filter,gaussian_filter1d
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

labelsize=18
myfontsize=labelsize
ticklabsize=14
cosmic_time, redshifts, log1pz, dt = get_time_redshift_bins()

#Get model_names for the interesting model variation
SFMR_ref='P23slope08'
calib='C20_ADF'#KK04b'#'C20_ADF'
delta_FMR_ZOH_asym_0=0.25
evolving_low_M_slope=True
lMmin1=6
lMmin2=8
SB_ref='Boco'#'Rinaldi24'#'Boco'
#SB_ref2='BiC18'
OFe_ref='pl400CCFep1'#'pl400CCFep1'#pl40CCFep03'#'pl400CCFep1'

cluster_data=False 

model_name = get_model_name(SFMR_ref=SFMR_ref, calib=calib, delta_FMR_ZOH_asym_0=delta_FMR_ZOH_asym_0,\
     evolving_low_M_slope=evolving_low_M_slope, OFe_ref=OFe_ref, FMR_slope=0.27,\
     SB_ref=SB_ref, IMF_ref='K01',lMmin=lMmin1, lMmax=12., z_start=10.,\
     any_addons='None')

model_names, model_OFe_color,SB_flag =\
    get_model_selection(OFe_var='all', OFe_color=['#e66101','#5e3c99','#01665e'],\
    no_GSMF_fix=True, no_asfr_1=False, no_SB=True, skip_excluded_by_lGRB=True)

### SETTINGS HERE ___________________________________    
color_by_OFe_relation,Z_MZR0_test=True, False    
O_profile = True
zvar='lower'
Fe_color,Fe_lw='saddlebrown',3.5
O_color,O_lw='k',2
median_profile=True
mean_profile=False

output_to_file=False
#__________________________

if(output_to_file):

    f=open('./extracted_data.dat','w')
    f.write(\
    "# models from Chruslinska+for ever in prep \n # log10(SFRD[z=0]) = -1.95 -- -1.8 in those models, SFRD(z) \propto (1+z)^3 holds for z<0.2 \n # First line: [Fe/H] array; solar log(Fe/H) + 12 = 7.5 \n # Following lines: (each line = one model variation) \n # distribution of stellar mass formed between z=0 and z=0.1 in [Fe/H] bins (harldy any evolution in 0.05<z<0.1 range) normalized to 1 \n # note: there is an additional systematic uncertainty not included here : this would shift the entire distribution by -0.2 dex/+ 0.1 dex in [Fe/H] due to z~0 absolute oxygen abundance scale choice \n #___________________________________________________________________________________________________\n")

    
fig1 = plt.figure(figsize=(9, 5))
ax1 = plt.subplot2grid((1, 1), (0, 0))
data_operation="subtract model2" #not used

if(zvar=='upper'):
    z_upper_cuts=[ 0.3]
    z_lower_cuts=[0. for zi in z_upper_cuts] 
    z_limit_varied=z_upper_cuts       
    ax1.set_title("[X/H] distribution of mass formed in stars between $z$=0 and $z_{x}$ \n",fontsize=labelsize)
    offset = 1.2
    flipped=False
elif(zvar=='lower'):
    z_lower_cuts=[0]#0.,1.,2.,5.,8.,10.]
    z_upper_cuts=[10. for zi in z_lower_cuts]
    z_limit_varied=z_lower_cuts    
    ax1.set_title("[X/H] distribution of mass formed in stars between $z_{x}$ and $z$=10\n",fontsize=labelsize)
    offset = 1.3
    flipped=True

xticks=[] 
only_once=True   
for i, (z_upper_cut, z_lower_cut) in enumerate(zip(z_upper_cuts, z_lower_cuts)):
    dx = offset * i
    if(z_upper_cut==z_lower_cut): dx-=offset*0.6
    xticks.append(dx)
    if(z_upper_cut==z_lower_cut):
            ax1.axvline(x=dx,lw=0.5,c='gray',ls='--')
    else:
        # Store for profiles from different models
        Fe_profiles = []
        O_profiles = []

        Fe_color_var=Fe_color
        for model_name, OFe_c in zip(model_names,model_OFe_color):
            # Fe profiles
            _, _, _, _, _, FeH_arr, FeH_z_integral = get_data_cut_in_redshift(
                metallicity_probe="FeH",
                z_lower_cut=z_lower_cut, z_upper_cut=z_upper_cut,
                model_name=model_name, data_path=data_path,
                cosmic_time=cosmic_time, redshifts=redshifts,
                cluster_data=cluster_data, data_operation=data_operation)

            if(only_once and output_to_file):
                for feh in FeH_arr:
                    f.write(str('%.5f'%(feh))+' ')
                f.write('\n')
                only_once=False
                
            norm = 1. / np.sum(FeH_z_integral)
            Fe_profiles.append(FeH_z_integral)
            plt.plot(FeH_arr,FeH_z_integral*norm, lw=1+i)
            if(output_to_file):           
                for feh in FeH_z_integral:
                    f.write(str('%.3f'%(feh*norm))+' ')
                f.write('\n')
        if(zvar=='lower'): zstr=z_lower_cut
        else: zstr=z_upper_cut
        plt.plot(FeH_arr,FeH_z_integral*norm, lw=1+i, label='$z_{x}$='+str(zstr)) 
            
plt.ylabel(r'$\rho_{*}$ [M$_{\odot}$ cMpc$^{-3}$]')
plt.xlabel('[Fe/H]')
plt.legend()
plt.show()
