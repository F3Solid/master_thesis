from dotenv import load_dotenv
#Make sure to create .env in your 'main_dir' including path to your main_dir
#   e.g. MY_PROJECT_PATH='/default/path/to/folder/'
#then keep the structure of the folders the same as on github and everything should work
import os
load_dotenv()
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
#from add_literature_datasets import *
from scipy.ndimage import gaussian_filter,gaussian_filter1d
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

labelsize=16
ticklabsize=12
cosmic_time, redshifts, log1pz, dt = get_time_redshift_bins()

#Get model_names for the interesting model variation
SFMR_ref='P23'
calib='C20_ADF'#KK04b'#'C20_ADF'
delta_FMR_ZOH_asym_0=0.25
evolving_low_M_slope=True
lMmin1=6
SB_ref='Boco'
OFe_ref='pl400CCFep1'#'pl400CCFep1'#pl40CCFep03'#'pl400CCFep1'
cluster_data=False 

model_name = get_model_name(SFMR_ref=SFMR_ref, calib=calib, delta_FMR_ZOH_asym_0=delta_FMR_ZOH_asym_0,\
     evolving_low_M_slope=evolving_low_M_slope, OFe_ref=OFe_ref, FMR_slope=0.27,\
     SB_ref=SB_ref, IMF_ref='K01',lMmin=lMmin1, lMmax=12., z_start=10.,\
     any_addons='None')

data_operation="subtract model2" #not used

z_upper_cuts=[0.075,1.,3.,6.,10.]
z_lower_cuts=[0. for zi in z_upper_cuts]
Fe_color,Fe_lw='saddlebrown',3
O_color,O_lw='k',1

fig1=plt.figure(figsize=(10,5.))
ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)

offset=1
i=0
O_profile=True
for z_upper_cut, z_lower_cut in zip(z_upper_cuts,z_lower_cuts):
    
    dx=offset*i
    SFRD_data, zz,lbts,_,_, XH_arr, FeH_z_integral=\
    get_data_cut_in_redshift(metallicity_probe="FeH",\
            z_lower_cut=z_lower_cut, z_upper_cut=z_upper_cut,\
            model_name=model_name,\
            data_path=data_path,\
            cosmic_time=cosmic_time,redshifts=redshifts,\
            cluster_data=cluster_data,\
            data_operation=data_operation)

    data=FeH_z_integral
    norm = 1./max(data)
    ax1.plot(dx+data*norm,XH_arr,c=Fe_color,lw=Fe_lw)
    if(O_profile):
        SFRD_data, zz,lbts,_,_, XH_arr, OH_z_integral=\
        get_data_cut_in_redshift(metallicity_probe="OH",\
                z_lower_cut=z_lower_cut, z_upper_cut=z_upper_cut,\
                model_name=model_name,\
                data_path=data_path,\
                cosmic_time=cosmic_time,redshifts=redshifts,\
                cluster_data=cluster_data,\
                data_operation=data_operation)                        

        data=OH_z_integral
        norm = 1./max(data)
        ax1.plot(dx+data*norm,XH_arr,c=O_color,lw=O_lw)
    i+=1   
if(O_profile):
    ax1.plot(-1,-10, c=O_color,lw=O_lw,label='X=O')
ax1.plot(-1,-10, c=Fe_color,lw=Fe_lw,label='X=Fe')

# Create custom x-tick positions and labels
xticks = [offset * i for i in range(len(z_upper_cuts))]
lookback_times = [cosmology.lookback_time(z).to(u.Gyr).value for z in z_upper_cuts]
xtick_labels = [f"{z:.2g}" for z in z_upper_cuts]  # format as desired
# Apply to the plot
ax1.set_xticks(xticks)
ax1.set_xticklabels(xtick_labels)
ax1.set_title("[X/H] distribution of mass formed in stars between $z$=0 and $z_{x}$ \n",fontsize=labelsize)
ax1.set_xlabel("redshift $z_{x}$", fontsize=labelsize)

secax = ax1.secondary_xaxis('top')#, functions=(forward, inverse))
secax.set_xlabel('Lookback Time [Gyr] at $z_{x}$', fontsize=labelsize)
secax.set_xticks(xticks)
secax.set_xticklabels([f"{lt:.1f}" for lt in lookback_times])

ax1.set_ylim([-3,1.5])
ax1.set_xlim([min(xticks)-0.5,max(xticks)+2])
secax.set_xlim([min(xticks)-0.5,max(xticks)+2])
ax1.set_ylabel('metallicity [X/H]', fontsize=labelsize)

ax1.legend(loc='upper right',fontsize=labelsize, frameon=False)
if(True):
        lw_sun=2
        c_sun='darkorange'
        xdata=[max(xticks)+2.2]
        ax1.axhline(y=0, lw=lw_sun, c=c_sun, ls='--')
        ax1.annotate(
                    'GS98 "solar" ratio',
                    xy=(0.79*max(xdata), 0.14), 
                color=c_sun,
                ha='left',
                rotation=0,
                va='top',
                zorder=6,
                fontsize=ticklabsize-2
                )
                                 
        ax1.axhline(y=-1., lw=lw_sun, c=c_sun, ls='--')            
        ax1.annotate(
                    '10% "solar"',
                    xy=(0.84*max(xdata), -0.85), 
                color=c_sun,
                ha='left',
                rotation=0,
                va='top',
                zorder=6,
                fontsize=ticklabsize-2                  
                )

        ax1.axhline(y=-0.8, lw=lw_sun, c='gray', ls='--',alpha=0.5)
        ax1.annotate(
                    ' SMC  [O/H]',
                    xy=(0.84*max(xdata), -0.8+0.13), 
                color='gray',
                ha='left',
                rotation=0,
                va='top',
                zorder=6,
                fontsize=ticklabsize-2
                )

        ax1.axhline(y=-0.47, lw=lw_sun, c='gray', ls='--',alpha=0.5)
        ax1.annotate(
                    ' LMC  [O/H]',
                    xy=(0.84*max(xdata), -0.47+0.13), 
                color='gray',
                alpha=0.5,
                ha='left',
                rotation=0,
                va='top',
                zorder=6,
                fontsize=ticklabsize-2
                )
        ax1.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=False, #top ticks off
#            labelbottom=False,
            labeltop=False,
            labelleft=True,
            labelright=False,
            labelsize=ticklabsize) # labels along the bottom edge are off
            
        secax.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=True, #top ticks off
#            labelbottom=False,
            labeltop=True,
            labelleft=True,
            labelright=False,
            labelsize=ticklabsize) # labels along the bottom edge are off

w,h=fig1.get_size_inches()
fig1.set_size_inches(w,h+2.2, forward=True)
ttop,lleft,rright,bbottom=0.82,0.1,0.94,0.1
plt.subplots_adjust(left=lleft, bottom=bbottom, right=rright, top=ttop, wspace=0.01, hspace=0.0)
    
plt.show()

