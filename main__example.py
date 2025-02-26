import pickle
import gc
import analysis_playground.analysis_tools as an 
from shockfind_interface import shock_finder
import numpy as np
import matplotlib.pyplot as plt
import os
if __name__=="__main__":
    names=[
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "magnetic_field_x",
            "magnetic_field_y",
            "magnetic_field_z",
            "pressure",
            "density",
             "dx"]
    level= 7
    analysis_name = "SNR_MHB"
    directory = "data/%s"%analysis_name + "/"
    try:
        os.makedirs(directory)
    except Exception as e: 
        print(e)
        pass 
    ### collecting data from simulations
    if 0:    
    
        #ds = an.analyse(outpath="/home/mattia/codes/CR/cr_particles_multip_MESS", outnumb=285)
        ds = an.analyse(outpath="/work/pacicco/simulations/SNR_paper/SNRs_uniform/SN_uniform_MB", outnumb=21)

        dx   = (ds.width/2**level)[0].in_cgs().d
        dens, cube = ds.get_cube(level=level, field="density", ghost=1)
        if level > 8:
            directory2 = directory+"many_lvl_%i/"%level
            try:
                os.makedirs(directory2)
            except Exception as e: 
                print(e)
                pass 
            for name in names:
                with open(directory2+"%s.pickle"%name, 'wb') as handle:
                    if name=="dx":
                        c=pickle.dump(dx, handle)      
                    else:
                        data,cube=ds.get_cube(level=9, field=name, ghost=1)
                        c=pickle.dump(data.in_cgs().d, handle)
                        del data,cube,c
                        gc.collect()
                        
        else:
            vx  =  cube["gas",  "velocity_x"].in_cgs().d
            vy  =  cube["gas",  "velocity_y"].in_cgs().d
            vz  =  cube["gas",  "velocity_z"].in_cgs().d
            Bx  =  cube["gas",  "magnetic_field_x"].in_cgs().d
            By  =  cube["gas",  "magnetic_field_y"].in_cgs().d
            Bz  =  cube["gas",  "magnetic_field_z"].in_cgs().d
            P   =  cube["gas",  "pressure"        ].in_cgs().d
            rho =  cube["gas",  "density"].in_cgs().d
            datas = vx,vy,vz,Bx,By,Bz,P,rho,dx
            #os.mkdir("pickled_data_from_SNR")
            with open(directory+"total_%i.pickle"%level, 'wb') as handle:
                  pickle.dump(datas, handle)
        quit()
    
    if level > 8:
        a=[]        
        for name in names:
            directory2 = directory+"many_lvl_%i"%level
            with open(directory2+"%s.pickle"%name, 'rb') as handle:
                    a.append(pickle.load( handle))
    else:
        with open(directory+"total_%i.pickle"%level, 'rb') as handle:
            a=pickle.load( handle)
    
    vx, vy, vz, Bx, By, Bz, P, rho, dx = a
    del a
    
    print("Data loaded")
    
#########################################################################################################
    
    shocksfinder=shock_finder()
    load = True
    if load:
        shocksfinder.load_results()
        
        
    else:
        # load data to run analysis 
        shocksfinder.load_data(rho,       # denisty
                        [vx,vy,vz], # velocity field
                        [Bx,By,Bz], # magnetic field 
                        P         , # presure
                        dx = dx     # spatial resolution - only for uniform grid
                        )
        # set code parameters
        shocksfinder.extra_params(periodic     = [ False ] * 3, # boundary condition 
                            gamma        = 1.666667 )     # ad index
        # define thresholds. Since use_gradTRho = True, they aren't really used in this case. 
        shocksfinder.set_thresholds(dx         = dx,            # spatial resolution 
                            vshock_min = 5e5,             # minimum velocity of a shock 
                            rhomean    = 3.e-24           # average pre-shock density 
                            )
        # find cells canditates for the shock searching algorithm 
        shocksfinder.find_candidates(use_gradTRho = True,       # use the Prfommer crityria to select shocks 
                               dx           = dx)         #
        # plot the locations of the shock candidates 
        #shocksfinder.plot_candidates()
        
        plt.show()
        
        # run analysis of te shock candidates
        shocksfinder.analyse_candidates(quiet=False)
        # collect resulting data, only shock position and firectional information  
        shocksfinder.shocks_data()
        # save the results to file for later inspection 
        shocksfinder.save_results()
    #finally:
    # make a 3D plot with all the shocks found
    #shocksfinder.plot3D(types="sf", alpha=0.1, ss=1)
    # make histograms of the various shock quantities
    #shocksfinder.histograms(9, bins=50, log=True, alpha=0.5)
    #print("done")
    shocksfinder_results= shocksfinder.load_results()
    
    shocks = shocksfinder_results[0]#.shocks
    cond   = np.logical_and(
                            np.logical_and(
                                            shocks[6] > 0,
                                            shocks[15]==0
                                          ),
                            shocks[14]==1
                           )
    header=shocksfinder_results[1][1]
    with open("%s_shocks+header_%d.pk"%(analysis_name,level), 'wb') as handle:
                  pickle.dump([shocks,header], handle)
    shocksfinder.plot3D()
    plt.show()
    quit()
