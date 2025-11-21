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
from scipy.ndimage import gaussian_filter,gaussian_filter1d
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
    
def prep_SFRD_redshift_metallicity_plot(ax1,SFRD_data,XH_arr,\
    xdata,x_label='redshift',onlycontour=False,\
    XH_SFR_weighted='none',\
    label_model='',c0='saddlebrown',lw0=2,ls0='-',\
    vmin=3e-5,vmax=0.25,levels=[0.01,0.05,0.1],\
    plot_contours=True,plot_maxima=True,cmap='viridis',lw_contours=4,lw_sun=2.,
    c_sun='darkorange',c_contours='saddlebrown',ticklabsize=14,labelsize=16,\
    label_solar=True,\
    threshold=None, only_threshold_line=False,show_scatter_threshold_line=False,\
    fit_threshold_line=True,upper_line=True,plot_percentiles=False, maxima_ref=None,\
    cbar_label=r'$\rm \frac{SFRD}{\Delta z \Delta [Fe/H]} [M_{\odot}/Mpc^{3}yr]$',\
    mark_MCs_FeH=True):
    
    ax1.set_xlabel(x_label, fontsize=labelsize+1)
    ax1.set_ylabel('metallicity [X/H]', fontsize=labelsize+1)  
    ax1.set_xlim([min(xdata), max(xdata)])

    ax11 = ax1.twinx()
    ax1.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=False, #top ticks off
#            labelbottom=False,
            labeltop=False,
            labelleft=True,
            labelright=False,
            labelsize=ticklabsize) # labels along the bottom edge are off
    ax11.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            top=False,
#            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=True,
            labelsize=ticklabsize) # labels along the bottom edge are off
    ax1.set_ylim([-3, 1.5])          
    ax11.set_ylim([-3, 1.5])              

    #print(np.max(SFRD_data))
    dXH_arr=np.array([])
    if(onlycontour==False and only_threshold_line==False):
        pcm=ax11.pcolormesh(xdata, XH_arr, np.transpose(SFRD_data),\
                 norm=colors.LogNorm(vmin=vmin, vmax=vmax),facecolor='none',cmap=cmap )#,cmap='OrRd' )      
        pcm=ax1.pcolormesh(xdata, XH_arr, np.transpose(SFRD_data),\
                 norm=colors.LogNorm(vmin=vmin, vmax=vmax),cmap=cmap )#,cmap='OrRd' ) 
                 
    if(threshold): 
        # Transpose back to original if needed, so we loop over columns
        data_T = SFRD_data#np.transpose(SFRD_data)  # shape: (len(xdata)-1, len(XH_arr)-1)    
        if(maxima_ref is not None):
            SFRD_atz_Z = np.array([SFRD_data[:][ii]/maxima_ref[ii] for ii in range(len(SFRD_data[:,0]))])
            data_T = SFRD_atz_Z
            #Normalize by the max SFRD/dzdZ(z) and apply the threshold to normalised array
            
        #threshold = 1e-2
        # Compute top edge below threshold
        top_edge_y = []
        for col in data_T:
            # Find indices where value is below the threshold
            below = np.where(col > threshold)[0]
#            else:
#                below = np.where(col < threshold)[0]
            if len(below) == 0:
                top_edge_y.append(np.nan)  # No value below threshold in this column
            else:
                if(upper_line): max_idx = max(below)  # last index that is still < threshold
                else: max_idx=min(below)
                top_edge_y.append(XH_arr[max_idx])

        # Crop xdata to match number of data columns
        x_centers = xdata#0.5 * (xdata[:-1] + xdata[1:])

        # Interpolate over NaNs
        x_valid = np.array(x_centers)[~np.isnan(top_edge_y)]
        y_valid = np.array(top_edge_y)[~np.isnan(top_edge_y)]

        interp_func = interp1d(x_valid, y_valid, kind='linear', fill_value='extrapolate')
        top_edge_y_interp = interp_func(x_centers)

        # Then apply smoothing
        smoothed_top_edge_y = gaussian_filter1d(top_edge_y_interp, sigma=2)
        # or use savgol_filter on top_edge_y_interp

        # --- Extrapolation settings ---
        x_fit_min = 3.   # use values where x > 1.0 for fitting
        x_extrap_max = 4  # extrapolate for x < 0.8

        # Select data to fit (clean, right-side region)
        fit_mask = x_centers > x_fit_min
        x_fit = x_centers[fit_mask]
        y_fit = smoothed_top_edge_y[fit_mask]

        # Linear fit (could also do log(x) or poly if needed)
        coeffs = np.polyfit(x_fit, y_fit, deg=1)
        linear_trend = np.poly1d(coeffs)

        # Create extrapolated region
        extrap_mask = x_centers < x_extrap_max
        x_extrap = x_centers[extrap_mask]
        y_extrap = linear_trend(x_extrap)

        # Merge: use extrapolated values where x < x_extrap_max
        final_top_edge = np.copy(smoothed_top_edge_y)
        final_top_edge[extrap_mask] = y_extrap

        # Plot the full smoothed + extrapolated line
#        ax1.plot(x_centers, final_top_edge, color=color, linewidth=6,alpha=0.5, label='')
        if(upper_line):        color,ls,alpha='w','-',0.6
        else:  color,ls,alpha='w','-',0.9
        if(fit_threshold_line):
            ax1.plot(x_centers, final_top_edge, color=color, linewidth=1.5,alpha=alpha,ls=ls,label='')        
        else:
            ax1.plot(x_centers, smoothed_top_edge_y, color=color, linewidth=1.5,ls=ls, label='')
        if(show_scatter_threshold_line): ax1.scatter(x_centers, top_edge_y, color='red', marker='o')

        if(upper_line):
            x_arrow=1.1
            y_arrow=linear_trend(x_arrow)
            ax1.annotate(r'contribute>10%', xy=(x_arrow+0.05,y_arrow+0.05),\
            xycoords='data',c=color,fontsize=14, alpha=1, rotation=-270, va='top') 
        else:
            x_arrow=0.6
            y_arrow= min(top_edge_y_interp[x_centers<=x_arrow])
#            ax1.annotate(' only in\n M$_{*}$<$10^{8}$M$_{\odot}$',\
            ax1.annotate('   M$_{*}$<$10^{8}$M$_{\odot}$\nonly',\
                xy=(x_arrow-0.4,y_arrow-0.5),\
            xycoords='data',c=color,fontsize=14, alpha=1, rotation=-270, va='bottom') 

        ax1.annotate(
                    '',
                    xy=(x_arrow,y_arrow-0.6), 
                    xytext=(x_arrow,y_arrow),
                    arrowprops=dict(color=color, shrink=0.0, width=0.5,\
                               headwidth=5, headlength=10, alpha=alpha),           
                ha='center',
                rotation=90,
                va='center',
                zorder=6    
                )        
        
        
    if(plot_contours and only_threshold_line==False):
            CS=ax1.contour(xdata,XH_arr,\
             np.transpose(SFRD_data),levels=levels,colors=(c_contours,),\
             linestyles=('-',),linewidths=(lw_contours,))
            ax1.clabel(CS, fmt = '%g', colors = c_contours, fontsize=15) #contour line labels
    if(only_threshold_line==False):            
        if(plot_maxima and XH_SFR_weighted=='none'):

                ls0='-'
                image0 = gaussian_filter(SFRD_data,1,0)
                SFRD_atz_Z = np.array([image0[:][ii] for ii in range(len(image0[:,0]))])
                maxima = np.array([SFRDi[2:].max() for SFRDi in SFRD_atz_Z ])
                indices=[np.where( np.abs(maxi-SFRDi[2:])==np.min(np.abs(maxi-SFRDi[2:])))[0][-1]\
                             for maxi,SFRDi in zip(maxima,SFRD_atz_Z)]
                XHmax=np.array(XH_arr[indices])+(XH_arr[1]-XH_arr[0])*0.5
                if(label_model!=''):    
                        ax1.plot(xdata,smooth(np.array(XHmax),10),lw=lw0,c=c0,ls=ls0,label=label_model)
                else:   ax1.plot(xdata,smooth(np.array(XHmax),10),lw=lw0,c=c0,ls='-')
        elif(XH_SFR_weighted!='none'):
            if(label_model!=''):  
                ax1.plot(xdata,XH_SFR_weighted,lw=lw0,c=c0,ls=ls0,label=label_model)
            else:
                ax1.plot(xdata,XH_SFR_weighted,lw=lw0,c=c0,ls=ls0)
            
        if(plot_percentiles):
            Y_perc_95 = compute_percentile_XH(SFRD_data, XH_arr, redshifts, percentile=0.95)
            Y_perc_5 = compute_percentile_XH(SFRD_data, XH_arr, redshifts, percentile=0.05)
            ax1.plot(xdata,Y_perc_95,lw=1,c=c0,ls=ls0)
            ax1.plot(xdata,Y_perc_5,lw=1,c=c0,ls=ls0)
                                        
    if(label_solar):
        ax1.plot(xdata, [0 for x in xdata], lw=lw_sun, c=c_sun, ls='--')
        ax1.annotate(
                    '"solar"',
                    xy=(0.88*max(xdata), 0.13), 
                color=c_sun,
                ha='left',
                rotation=0,
                va='top',
                zorder=6,
                fontsize=ticklabsize
                )
                
#        ax1.plot(xdata, [-0.14 for x in xdata], lw=lw_sun-1, c=c_sun, ls=':')
#        ax1.annotate(
#                    'A09 "solar" O ratio',
#                    xy=(0.8*max(xdata), -0.05), 
#                color=c_sun,
#                ha='left',
#                rotation=0,
#                va='top',
#                zorder=6,
#                fontsize=ticklabsize-4
#                ) 
                 
        ax1.plot(xdata, [-1. for x in xdata], lw=lw_sun, c=c_sun, ls='--')            
        ax1.annotate(
                    '10% "solar"',
                    xy=(0.8*max(xdata), -0.87), 
                color=c_sun,
                ha='left',
                rotation=0,
                va='top',
                zorder=6,
                fontsize=ticklabsize                  
                )

        ADF=0.2
        LMC_OH,SMC_OH=-0.47, -0.82
        if(calib=='C20_ADF'):
            LMC_OH, SMC_OH=LMC_OH+ADF, SMC_OH+ADF
        ax1.plot(xdata, [SMC_OH for x in xdata], lw=lw_sun, c='lightgray', ls='--',alpha=0.5)
        ax1.annotate(
                    ' SMC  [O/H]',
                    xy=(0.8*max(xdata), SMC_OH+0.15), 
                color='lightgray',
                ha='left',
                rotation=0,
                va='top',
                zorder=6,
                fontsize=ticklabsize-2
                )
        ax1.plot(xdata, [LMC_OH for x in xdata], lw=lw_sun, c='lightgray', ls='--',alpha=0.5)
        ax1.annotate(
                        ' LMC  [O/H]',
                        xy=(0.8*max(xdata), LMC_OH+0.15), 
                    color='lightgray',
                    alpha=0.5,
                    ha='left',
                    rotation=0,
                    va='top',
                    zorder=6,
                    fontsize=ticklabsize-2
                    )

        if(mark_MCs_FeH):
            #ax1.fill_between(xdata, [-1 for x in xdata], [-0.5 for x in xdata],\
            #edgecolor='brown', facecolor='None',hatch='/',alpha=0.15)
            trans = ax1.get_yaxis_transform() # x in data untis, y in axes fraction
            c=c0
            ax1.annotate(
                        'SMC [Fe/H]',
                    xy=(1.11, -0.45), 
                    color=c,
                    ha='left',
                    rotation=90,
                    va='top',
                    zorder=6,
                    fontsize=ticklabsize-2,
                    xycoords=trans
                    )
            ax1.annotate(
                        '',
                        xy=(1.1, -0.5), 
                        xytext=(1.1, -1),
                        arrowprops=dict(color=c, shrink=0.0, width=0.5,\
                                   headwidth=4, headlength=10, alpha=1),           
                    ha='left',
                    rotation=90,
                    va='center',
                    zorder=6 ,
                    xycoords=trans                   
                    )
            ax1.annotate(
                        '',
                        xy=(1.1, -1), 
                        xytext=(1.1, -0.5),
                        arrowprops=dict(color=c, shrink=0.0, width=0.5,\
                                   headwidth=4, headlength=10, alpha=1),           
                    ha='left',
                    rotation=90,
                    va='center',
                    zorder=6 ,
                    xycoords=trans                   
                    )

            LMC_Fe_bounds=[-0.5,-0.1]
            ax1.annotate(
                        'LMC [Fe/H]',
                    xy=(1.08, -0.05), 
                    color=c,
                    ha='left',
                    rotation=90,
                    va='center',
                    zorder=6,
                    fontsize=ticklabsize-2,
                    xycoords=trans
                    )
            ax1.annotate(
                        '',
                        xy=(1.07, -0.48), 
                        xytext=(1.07, -0.1),
                        arrowprops=dict(color=c, shrink=0.0, width=0.5,\
                                   headwidth=4, headlength=10, alpha=1),           
                    ha='left',
                    rotation=90,
                    va='center',
                    zorder=6 ,
                    xycoords=trans                   
                    )
            ax1.annotate(
                        '',
                        xy=(1.07, -0.1), 
                        xytext=(1.07, -0.5),
                        arrowprops=dict(color=c, shrink=0.0, width=0.5,\
                                   headwidth=4, headlength=10, alpha=1),           
                    ha='left',
                    rotation=90,
                    va='center',
                    zorder=6 ,
                    xycoords=trans                   
                    )


    ttop,lleft,rright,bbottom=0.87,0.12,0.84,0.09
    plt.subplots_adjust(left=lleft, bottom=bbottom, right=rright, top=ttop, wspace=0.01, hspace=0.0)
    if(onlycontour==False and only_threshold_line==False):
        cbaxes = fig1.add_axes([lleft, ttop, rright-lleft, 0.02])  
        # This is the position for the colorbar left bottom width height
        cb=plt.colorbar(pcm, orientation='horizontal', cax=cbaxes)
        cb.set_label(label=cbar_label,\
                     size=labelsize, labelpad=15)    

        cb.ax.xaxis.set_ticks_position('top')
        cb.ax.xaxis.set_label_position('top')
        cb.ax.tick_params(labelsize=ticklabsize)
    
    return ax1

#fill in the time-redshift arrays (as used to calculate the models)
cosmic_time, redshifts, log1pz, dt = get_time_redshift_bins()

#Get model_names for possible model variations
SFMR_ref_list=['P23slope08','P23','P23_modif_highz']
calib_list=['C20_ADF']
evolving_low_M_slope_list=[True,False]
#MZR_evol_list=[True,False]
OFe_ref_list=['pl400CCFep1','pl40CCFep03']
FMR_slope_list=[0.27]
SB_ref_list=['Boco','Rinaldi24']

model_list,_ = get_input_models(SFMR_ref_list=SFMR_ref_list,\
             calib_list=calib_list,evolving_low_M_slope_list=evolving_low_M_slope_list,\
             SB_ref_list=SB_ref_list,OFe_ref_list=OFe_ref_list,\
             FMR_slope_list=FMR_slope_list,full_length=len(redshifts), data_path=data_path)
             
#Get model_names for the interesting model variation
#SFR-P23slope08-Z-C20_ADF-Zev-True-GSMFev-True-OFe-G05CCFep07-FMR0-27-SB-Boco-dFMR3_10-25
#SFR-P23-Z-C20_ADF-Zev-True-GSMFev-True-OFe-G05CCFep07-FMR0-27-SB-Boco-dFMR3_10-25
SFMR_ref='P23_modif_highz'
calib='C20_ADF'#KK04b'#'C20_ADF'
delta_FMR_ZOH_asym_0=0.25
evolving_low_M_slope=True
lMmin1=6
lMmin2=8
SB_ref='Boco'#'Rinaldi24'#'Boco'
#SB_ref2='BiC18'
OFe_ref='G05CCFep07'
#OFe_ref='pl40CCFep03'#pl40CCFep03'#'pl400CCFep1'
cluster_data=False 
cluster_str='form'
#cluster_ref=create_cluster_ref(slope_cluster_MF=2, lMmin_cluster=3.5, nbin_cluster=13)
#slope_cluster_MF, lMmin_cluster, nbin_cluster = extract_cluster_info(cluster_ref)
cluster_ref='None'
bin_val=10**5
cluster_mass_binning=None#[10**lMmin_cluster,1e9,nbin_cluster]
cluster_mass_binning_desc=None#['min value', 'max value','number of bins']
cluster_mass_bins = None#get_cluster_mass_bins(cluster_mass_binning=cluster_mass_binning,\
#        cluster_mass_binning_desc=cluster_mass_binning_desc)
        
#bin_idx=find_bin_index(bin_edges=cluster_mass_bins, value=10**4)
#print(bin_idx)

model_name = get_model_name(SFMR_ref=SFMR_ref, calib=calib, delta_FMR_ZOH_asym_0=delta_FMR_ZOH_asym_0,\
     evolving_low_M_slope=evolving_low_M_slope, OFe_ref=OFe_ref, FMR_slope=0.27,\
     SB_ref=SB_ref, IMF_ref='K01',lMmin=6., lMmax=12., z_start=10.,cluster_ref=cluster_ref,\
     any_addons='None')

model_name2 = get_model_name(SFMR_ref=SFMR_ref, calib=calib, delta_FMR_ZOH_asym_0=delta_FMR_ZOH_asym_0,\
     evolving_low_M_slope=evolving_low_M_slope, OFe_ref=OFe_ref, FMR_slope=0.27,\
     SB_ref=SB_ref, IMF_ref='K01',lMmin=lMmin2, lMmax=12., z_start=10.,cluster_ref=cluster_ref,\
     any_addons='None')     

if(delta_FMR_ZOH_asym_0!=0.): Z_label=' FMR0(evol. at z>3)= '+str(calib)
else: Z_label=' FMR0= '+str(calib)
if(evolving_low_M_slope): GSMF_label=r' $\alpha_{\rm GSMF}(z)$'
else: GSMF_label=r' $\alpha_{\rm GSMF}=\alpha_{\rm fix}$'
if(OFe_ref=='pl400CCFep1'): OFe_label='"Fe-poor"'
elif(OFe_ref=='pl40CCFep03'): OFe_label='"Fe-rich"'
else: OFe_label=OFe_ref
title_label='SFMR='+SFMR_ref+Z_label+GSMF_label+' O->Fe '+OFe_label
    
##______OR simply set model_name by hand
title_label=''

print(model_name, type(model_name))
data_operation="subtract model2"
#__________PLOT______________________________________________
fig1=plt.figure(figsize=(8,6))
ax1 = plt.subplot2grid((1, 1), (0, 0), colspan=1, rowspan=1)
## use iron-abundance as a metallicity probe metallicity_probe="FeH"
#fraction_of_SB_SF=0.5,data_operation="subtract model2"
data_operation="subtract model2"
levels=[0.01]
plot_contours=True
cOH,cFe='k','chocolate'#saddlebrown'

Fe_colored=True
if(Fe_colored):
    add_lines_below_logM8=False
    metallicity_probe='FeH'
    onlycontour_Fe=False
    onlycontour_O=True
    cbar_label=r'$\rm \frac{SFRD}{\Delta z \Delta [Fe/H]} [M_{\odot}/Mpc^{3}yr]$'
    cbar_label=r'$\rm SFRD([Fe/H],z) \ [M_{\odot} \ Mpc^{-3} \ yr^{-1}]$'    
    mark_MCs_FeH=True
    lw_Fe_avg = 5
    lw_O_avg = 3
else:
    add_lines_below_logM8=False
    metallicity_probe='OH'
    onlycontour_Fe=True
    onlycontour_O=False
    cbar_label=r'$\rm \frac{SFRD}{\Delta z \Delta [O/H]} [M_{\odot}/Mpc^{3}yr]$'
    mark_MCs_FeH=False
    lw_Fe_avg = 5
    lw_O_avg = 3
        
SFRD_data, SFRD_z_tot,_,_,_,_, XH_arr,FeH_SFRD_weighted=\
            get_data_cut_in_metallicity(metallicity_probe="FeH",\
            XH_lower_cut=-5,XH_upper_cut=5,\
            model_name=model_name,\
            data_path=data_path,\
            cosmic_time=cosmic_time,redshifts=redshifts,\
            cluster_data=cluster_data,cluster_str=cluster_str,\
            bin_val=bin_val,\
            cluster_mass_binning=cluster_mass_binning,\
            cluster_mass_binning_desc=cluster_mass_binning_desc,data_operation=data_operation)

image0 = gaussian_filter(SFRD_data,1,0)
SFRD_atz_Z = np.array([image0[:][ii] for ii in range(len(image0[:,0]))])
maxima_ref = np.array([SFRDi[2:].max() for SFRDi in SFRD_atz_Z ])
                
prep_SFRD_redshift_metallicity_plot(ax1=ax1,XH_arr=XH_arr,SFRD_data=SFRD_data,\
    XH_SFR_weighted=FeH_SFRD_weighted,\
    xdata=redshifts[::-1],x_label='redshift',onlycontour=onlycontour_Fe,\
    label_model=r'<[Fe/H]>$_{\rm SFRD}$',c0=cFe,lw0=lw_Fe_avg,ls0='-',lw_contours=3,\
    levels=levels,plot_contours=plot_contours,c_contours=cFe,\
    cbar_label=cbar_label,mark_MCs_FeH=mark_MCs_FeH)    
#    label_model='X=Fe',c0='saddlebrown',lw0=4,ls0='-',lw_contours=2)

## oxygen-abundance as a metallicity probe   -- plot only contours, for comparison    
SFRD_data, SFRD_z_tot,_,_,_,_, XH_arr,XH_SFRD_weighted=\
            get_data_cut_in_metallicity(metallicity_probe="OH",\
            XH_lower_cut=-5,XH_upper_cut=5,\
            model_name=model_name,\
            data_path=data_path,\
            cosmic_time=cosmic_time,redshifts=redshifts,\
            cluster_data=cluster_data,cluster_str=cluster_str,\
            bin_val=bin_val,\
            cluster_mass_binning=cluster_mass_binning,\
            cluster_mass_binning_desc=cluster_mass_binning_desc, info_print=True,data_operation=data_operation)

prep_SFRD_redshift_metallicity_plot(ax1=ax1,XH_arr=XH_arr,SFRD_data=SFRD_data,\
    XH_SFR_weighted=XH_SFRD_weighted,\
    xdata=redshifts[::-1],x_label='redshift',onlycontour=onlycontour_O,\
    label_model=r'<[O/H]>$_{\rm SFRD}$',c0=cOH,lw0=lw_O_avg,ls0='-',c_contours=cOH,\
    lw_contours=1.5,levels=levels,plot_contours=plot_contours,cbar_label=cbar_label,mark_MCs_FeH=mark_MCs_FeH)


if(title_label!=''): 
        ax1.annotate(
                title_label,
                xy=(0.05, 1.45), 
            color='darkorange',
            ha='left',
            rotation=0,
            va='top',
            zorder=6    
            ) 

ax1.legend(loc='upper right', fontsize=13)
w,h=fig1.get_size_inches()
fig1.set_size_inches(w+0.5,h+2., forward=True)
#plt.savefig(tools_dir_path+'/plots/SFRD_ZZ_z/'+model_name+'.png')
plt.show()
