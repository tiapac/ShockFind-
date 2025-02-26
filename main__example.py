import pickle
import gc
import analysis_playground.analysis_tools as an 
import  codes.shockfinder as css
shock_finder = css.shock_finder
import numpy as np
import matplotlib.pyplot as plt
import os

level= 8
analysis_name = "SNR_HB"
analysis_name += "lvl_%02d"%level
outnumb = 12

directory = "data/%s_%05d"%(analysis_name,outnumb) + "/"
ncpus = 16
rescale=0
load = False
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
    
   
    ### collecting data from simulations
    def loadytData():  
        def fromScratch():
            try:
                os.makedirs(directory)
            except Exception as e: 
                print(e)
                pass 
            print("Computing from scratch")
            #ds = an.analyse(outpath="/home/mattia/codes/CR/cr_particles_multip_MESS", outnumb=285)
            ds = an.analyse(outpath="/work/pacicco/simulations/SNR_paper/SNRs_uniform/SN_uniform_MB", outnumb=outnumb)

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
                datas = [vx,vy,vz,Bx,By,Bz,P,rho,dx]
                #os.mkdir("pickled_data_from_SNR")
                with open(directory+"total_%i.pickle"%level, 'wb') as handle:
                    pickle.dump(datas, handle)
                return datas
        try:
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
            return [vx, vy, vz, Bx, By, Bz, P, rho, dx]
        except Exception as e:
            return fromScratch()    
        
    
#########################################################################################################
    
    shocksfinder=shock_finder(name=analysis_name)
    
    if load:
        shocksfinder.load_results(path=directory, name=analysis_name)
  
        
    else:
        vx, vy, vz, Bx, By, Bz, P, rho, dx =  loadytData()  
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
        
        #plt.show()
        
        # run analysis of te shock candidates
        shocksfinder.analyse_candidates(quiet=False, ncpus = ncpus, rescale=rescale)
        # collect resulting data, only shock position and firectional information  
        shocksfinder.shocks_data()
        
        # save the results to file for later inspection 
        shocksfinder.save_results(path=directory, name=analysis_name)
    #finally:
    # make a 3D plot with all the shocks found
    #shocksfinder.plot3D(types="sf", alpha=0.1, ss=1)
    # make histograms of the various shock quantities
   
    names = {i:name for i,name in enumerate(shocksfinder.header[1])}
    #print(names)
    shared = dict(bins=15 , log=True , alpha=0.8, histtype="step", lw=2, density=True)
    ax, fig = shocksfinder.histograms(9, **shared )
    shocksfinder.histograms(10, ax = ax, fig=fig, linestyle="--", **shared )
    #print("done")
    shocksfinder_results= shocksfinder.results#shocksfinder.load_results(path=directory, name=analysis_name)

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
