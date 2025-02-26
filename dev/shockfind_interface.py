import numpy as np
import matplotlib.pyplot as plt
from shockfind.shockfind import core
import pickle
import analysis_playground.analysis_tools as an 
import utils 
import time
import pyvista as pv
import gc
import os 
#class data_onj:
#    def __init__(self) -> None:
#        pass    
    

def set_periodic(f,g):
    n=len(f)-1
    
    # z-axis
    
    g_i_minus_1=g[:,:,n-1]
    g_i_plus_1 =g[:,:,0  ]
    fi         =g_i_minus_1-g_i_plus_1
    fi         /= 2.0
    
    f[:,:,n]=fi
    
    g_i_minus_1=g[:,:,n]
    g_i_plus_1 =g[:,:,1 ]
    fi         =g_i_minus_1-g_i_plus_1
    fi         /= 2.0
    
    f[:,:,0]=fi
    
    # y axis 
    g_i_minus_1=g[:,n-1,:]
    g_i_plus_1 =g[:,0,  :]
    fi         =g_i_minus_1-g_i_plus_1
    fi         /= 2.0
    
    f[:,n,:]=fi
    
    g_i_minus_1=g[:,n,:]
    g_i_plus_1 =g[:,1,:]
    fi         =g_i_minus_1-g_i_plus_1
    fi         /= 2.0
    
    f[:,0,:]=fi
    
    # x axis 
    g_i_minus_1=g[n-1,:,:]
    g_i_plus_1 =g[0,  :,:]
    fi         =g_i_minus_1-g_i_plus_1
    fi         /= 2.0
    
    f[n,:,:]=fi
    
    g_i_minus_1=g[n-1,:,:]
    g_i_plus_1 =g[1,  :,:]
    fi         =g_i_minus_1-g_i_plus_1
    fi         /= 2.0
    
    f[0,:,:]=fi

    return f

class shock_finder(core):
                    
                    
    def __init__(self,
                mhd:bool = False,
                use_pressure = True, name="shocks") -> None:
        if not mhd and not use_pressure:
            use_pressure = True
        self.name = name
        super().__init__(hydro_only = not mhd, use_pressure = use_pressure)
        
        #self.Rho      = None
        #self.divV     = None
        #self.P        = None
        #self.nablaRho = None
        #self.V        = None
        #self.B        = None
        #self.convergenge_threshold, self.nablaRho_threshold = [None]*2
        
        self.extra_params()
        pass        
        
    def load_data(self,
                Rho,
                V,
                B,
                P            = None,
                divV         = None,
                nablaRho     = None,
                dx           = None,
                )-> None:
        """This function is used to load the necessary data for the shock finding algorithm

        Args:
            Rho (3D array) : 3D cube with density values.
            V (3* 3D array): 3 3D cubes with velocity values vx,vy,vz.
            B (3* 3D array): 3 3D cubes with magnetic field values bx,by,bz.
            P (3D array)   : 3D cube with density values. 
            divV (3D array, optional): 3D cube with velocity divergence values. 
                                       If None, it is computed runtime.
                                       Defaults to None.
            nablaRho (3 3D array, optional): 3 3D cubes of the density gradients.
                                             If None, it is computed runtime.
                                             Defaults to None.

        Raises:
            Exception: If dx is not supplied and either divV or nablaRho are to be computed runtime.

        Returns:
            None
        """
        self.Rho         = Rho
        self.P           = P
        self.V           = V 
        self.B           = B 
        if divV is None: 
            if dx is None: 
                raise Exception("If divV is not supplied, \
                you must input a dx to compute the divergence runtime.")
            divV = self.divergence(V, dx=dx)
        self.divV        = divV
        if nablaRho is None: 
            if dx is None: 
                raise Exception("If density gradient is not supplied, \
                you must input a dx to compute the divergence runtime.")
            nablaRho  = np.gradient(Rho, dx,edge_order=2)
                
        self.nablaRho    = nablaRho
        
        return  
     
    def extra_params(self,
                    method_norm  = "point_gradient",
                    periodic     = [True,True,True],
                    Rgrad        = 3,
                    Rcylinder    = 3,
                    gamma        = 5./3.,
                    line_range   = 10,
                    method_plane = "point_field",
                    field_ref    = 0,
                    shock_ratio  = 1.1,
                    offset       = [0,0,0]
                    ):
        """This function defines a set of parameters used as "extra" in SHOKFIND.
            It is used to both initialise and change the parameters. 

        Args:
            method_norm (str, optional): Method to be used to find shock normals.
                                         Defaults to "point_gradient".
            periodic (list, optional): Set what axis have periodic boundaries. 
                                        Defaults to [True,True,True].
            Rgrad (int, optional): _description_. Defaults to 3.
            Rcylinder (int, optional): Radius of the cylinder used to analyse the shock's profiles.
                                        Defaults to 3.
            gamma (float, optional): Adiabatic index of the gas. 
                                        Defaults to 5./3..
            line_range (int, optional): Lenght of the cylinder built to anaylse the shock's profile. Defaults to 10.
            method_plane (str, optional): _description_. Defaults to "point_field".
            field_ref (int, optional): _description_. Defaults to 0.
            shock_ratio (float, optional): _description_. Defaults to 1.1.
            offset (list, optional): _description_. Defaults to [0,0,0].
        return:
            extra (dict): the list of parameter setted.
        """
        extra = {"method_norm": method_norm,
                 "periodic":    periodic,
                 "Rgrad":       Rgrad,
                 "Rcylinder":   Rcylinder,
                 "gamma":       gamma,
                 "line_range":  line_range,
                 "method_plane":method_plane,
                 "field_ref":   field_ref,
                 "shock_ratio": shock_ratio,
                 "offset":      offset
                }
        self.extra = extra
        
        return extra
    
    def set_thresholds(self,
                       convergenge_threshold = None,
                       nablaRho_threshold    = None,
                       #############################
                       dx                    = None,
                       vshock_min            = None,
                       rhomean               = None,
                       #############################
                       N                     = 3   ,
                       compress              = 1.1   ,
                       
                       ):
        """ This funtion set the thresholds used in the selection of candidates for the shock
            finding algorithm. If both are None, then dx, vshock_min and rhomean must be supplied
            for an estimation.
            #######If only one is None and the other is supplied  , 
        Args:
            convergenge_threshold (float|str, optional): Convergence threshold value. 
                                                     Defaults to None.
            nablaRho_threshold (float|str, optional): Density gradient value. 
                                                  Defaults to None.
            dx (float, optional): Spatial resolution of the grid. 
                                  Defaults to None.
            vshock_min (float, optional): Minimum velocity of the shock used to compute the. 
                                          convergence threshold.
                                          Defaults to None.
            rhomean (float, optional): Average pre-shock density. 
                                       Defaults to None.
            N (int, optional): Number of cells of shock spreading. 
                               Defaults to 3.
            compress (int, optional): Shock density compression ration.
                                      Defaults to 4.

        Raises:
            Exception: If necessary parameters are missing.

        Returns:
            list: Return the values of the thresholds, convergenge_threshold, nablaRho_threshold and save them.
        """
        if convergenge_threshold is None and nablaRho_threshold is None: 
            if dx is None or vshock_min is None or rhomean is None: 
                raise Exception("If you do not directly specify thresholds,   \
                    then you must supply a minimum velocity for the shock     \
                    (vshock_min), the average pre-shock density (rhomean) and \
                    the separation between cells (dx)."
                                )
        if convergenge_threshold is None:
            if nablaRho_threshold is not None:
                self.convergenge_threshold = 0.0 
            else:  
                self.convergenge_threshold = (vshock_min*(compress-1.0)/(compress * N)) / dx
        if nablaRho_threshold is None: 
            if convergenge_threshold is not None:
                self.nablaRho_threshold    = 0.0
            else:
                self.nablaRho_threshold    = (rhomean)*(compress-1.0)/(N * dx)
            
        return self.convergenge_threshold, self.nablaRho_threshold

    def find_candidates(self,use_gradTRho = False, dx = None, kB = 1.3806490e-16, mu_mol = 1.2195e0, mh = 1.6605390e-24):
        """This function determines shock candidates based on the threshold setter in 
            set_threshold.

        Returns:
            list: list of the shock candidates index-wise locations.
        """
        self.shock_candidates = []
        nablaRho_mag = (self.nablaRho[0]**2 + self.nablaRho[1]**2 + self.nablaRho[2]**2)**0.5
        if use_gradTRho:
            if self.P is None: raise Exception("Cannot use this the gradT-gradRho critiria without the pressure.")
            if dx is None: 
                raise Exception("If gradTrho  is True, \
                you must input a dx to compute the pressure gradient runtime.")
            #nablaP=np.gradient(self.P, dx,edge_order=2)
            #nablaP_mag= (nablaP[0]**2 + nablaP[1]**2+nablaP[2]**2)**0.5
            print("computing dot product between temperature and density gradients... may require a bit.")
            T = (mu_mol * mh / kB) * self.P / self.Rho
            gradT    = np.gradient(T, dx,edge_order=2)
            gradTrho = gradT[0]*self.nablaRho[0]+gradT[1]*self.nablaRho[1]+gradT[2]*self.nablaRho[2]      
            cs       = np.sqrt(self.extra["gamma"]*self.P/self.Rho)
            bmag     = np.sqrt(self.B[0]**2+self.B[1]**2+self.B[2]**2)
            ca       = bmag / (np.sqrt(4.0*np.pi*self.Rho))
            vmag     = (self.V[0]**2+self.V[1]**2+self.V[2]**2)**0.5
            mach     = vmag / (0.7*np.minimum(cs,ca))
        else:
            gradTrho = None
            mach     = None
        if 0: 
            shocks  =  self.find_shocks( conv          = -self.divV,
                                                        grad          = nablaRho_mag,
                                                        threshold_grad= self.nablaRho_threshold,
                                                        conv_threshold= self.convergenge_threshold,
                                                        Ncells        = 1e5,
                                                        block         = 1)         
            x, y, z , _, _=  shocks[0]
            
        if 1: 
            shocks  =  self.find_shocks_simple(  conv           =  -self.divV,
                                                                grad           =  self.nablaRho_threshold,
                                                                conv_threshold = self.convergenge_threshold,
                                                                grad_threshold = self.nablaRho_threshold,
                                                                gtgrho         = gradTrho,
                                                                mach           = mach
                                                           )
            x, y, z =  shocks[0] #if not use_gradTRho else shocks[2]
        for i,_ in enumerate(shocks[0][0]):
            self.shock_candidates.append((x[i],y[i],z[i]))
        del nablaRho_mag
        if use_gradTRho: del  T, gradT, gradTrho, cs, bmag, ca, vmag, mach
        gc.collect()
        return self.shock_candidates 
          
    def analyse_candidates(self,quiet = False):
        """This funtion analyse the candidates shock cells and
            finds characterise the type of shocks present in the cells, if any.

        Returns:
            list: return the full set of information about the shoks and a header for the data.
        """
         
            #del self.shock_candidates[1::2]
            #del self.shock_candidates[1::2]
            #del self.shock_candidates[1::2]
            #del self.shock_candidates[1::2]
            #del self.shock_candidates[1::2]
            #del self.shock_candidates[1::2]
            #del self.shock_candidates[1::2]
            #del self.shock_candidates[1::2]
        #del self.shock_candidates[1::2] 
        #del self.shock_candidates[1::2] 
        #del self.shock_candidates[1::2]    
        t0=time.time()
        charact_funtion = self.characterise_shocks_para
        self.shocks, self.header = charact_funtion( self.shock_candidates,
                                                    self.Rho,
                                                    self.P,
                                                    self.B,
                                                    self.V,
                                                    self.divV,
                                                    self.nablaRho,
                                                    self.extra,
                                                    quiet = quiet,
                                                    ID='', ncpus=16)
        #print(self.shocks)
        print("took ",time.time() - t0, "seconds")
        return self.shocks, self.header
    
    def shocks_data(self, dx = 1.0):
        """This funtion quickly extract only the fast and slow shocks flagged with no problem
            and all the undefined shocks.

        Args:
            dx (float, optional): cell separation. Defaults to 1.

        Returns:
            list: 3D arrays of the shocks locations and pointers for fast and slow types.
        """
        ss = self.shocks
        condF   = np.logical_and(ss[6]==12, ss[15]==0  )
        condS   = np.logical_and(ss[6]==34, ss[15]==0  )#True )#
        condboh = np.logical_and(ss[6]==0 , True )

        xsF,ysF,zsF       = ss[0][condF]  , ss[1][condF]  , ss[2][condF]
        xsS,ysS,zsS       = ss[0][condS]  , ss[1][condS]  , ss[2][condS]
        xsboh,ysboh,zsboh = ss[0][condboh], ss[1][condboh], ss[2][condboh]
        ##vectors
        nxsS,nysS,nzsS= ss[3][condS],ss[4][condS],ss[5][condS]
        nxsF,nysF,nzsF= ss[3][condF],ss[4][condF],ss[5][condF]
        ###
        
        
        Fshocks     = [xsF   * dx,ysF   * dx,zsF   * dx]
        Sshocks     = [xsS   * dx,ysS   * dx,zsS   * dx]
        Bohshocks   = [xsboh * dx,ysboh * dx,zsboh * dx]
        ###
        Fpointers=[nxsS, nysS, nzsS]
        Spointers=[nxsF, nysF, nzsF]
        self.computed_shocks = [Fshocks, Sshocks, Bohshocks, Fpointers, Spointers]
        return self.computed_shocks, self.header
    
    def save_results(self, name=None):
        if name is None: name = self.name 
        print("Saving results to",name )
        with open("%s_result.pk"%name, 'wb') as handle:
                  pickle.dump([self.shocks,self.header], handle)
        return
    def load_results(self, name = None):
        if name is None: name = self.name 
        with open("%s_result.pk"%name, 'rb') as handle:
            self.shocks,self.header = pickle.load( handle)
        self.shocks_data()
        return self.shocks,self.header
    
    def plot_candidates(self, ax = None, fig = None, alpha = 0.5, ss = 2):
        if fig == None: fig=plt.figure(figsize=(8,8))
        if ax  == None: ax=fig.add_subplot(1, 1, 1, projection = '3d')
        xs,ys,zs=[],[],[]
        for s in self.shock_candidates:
            
            xs.append(s[0])
            ys.append(s[1])
            zs.append(s[2])
        p =ax.scatter(xs,ys,zs, alpha = alpha, s = ss, color = "midnightblue" )
        
        utils.set_axes_equal(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        
        return ax, fig  
    
    def plot3D(self, ax = None, fig = None, alpha = 0.5, ss = 2, types = "fs?"):
        types=types.upper()
        if fig == None: fig = plt.figure(figsize=(8,8))
        if ax  == None: ax  = fig.add_subplot(111, projection = '3d')
        Fshocks,Sshocks,Bohshocks,Fpointers,Spointers=self.computed_shocks
        if "F" in types:
            xsF,ysF,zsF=Fshocks
            p =ax.scatter(xsF,ysF,zsF, alpha=alpha, s=ss, color="midnightblue",label="FAST-shocks" )
        if "S" in types:
            xsS,ysS,zsS=Sshocks
            p =ax.scatter(xsS,ysS,zsS, alpha=alpha, s=ss, color="red",label="SLOW-shocks")
        if "?" in types:
            xsboh,ysboh,zsboh = Bohshocks
            p =ax.scatter(xsboh,ysboh,zsboh, alpha=alpha, s=ss, color="green",label="???-shocks")
        ax.legend()

        utils.set_axes_equal(ax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    

        return ax, fig
    
    def histograms(self,ind, ax = None, fig = None,  **kwargs ):
        if fig == None: fig = plt.figure(figsize=(8,8))
        if ax  == None: ax  = fig.add_subplot(111)
        ss = self.shocks
        header=self.header
        
        condF   = np.logical_and(ss[6]==12, ss[15]==0  )
        condS   = np.logical_and(ss[6]==34, ss[15]==0  )
        condboh = ss[6]==0 
        ax.hist(ss[ind][condF],  color="midnightblue", label="Fast n = %5d"%len(ss[ind][condF])  ,**kwargs )
        ax.hist(ss[ind][condS],  color="red",          label="Slow n = %5d"%len(ss[ind][condS])  ,**kwargs )
        ax.hist(ss[ind][condboh],color="green",        label="???  n = %5d"%len(ss[ind][condboh]),**kwargs )
        ax.set_xlabel(header[1][ind])
        ax.legend()
        return ax, fig 
    @staticmethod    
    def divergence(V, dx):
        vx,vy,vz = V
        dvxdx, dvydy, dvzdz = np.gradient(vx, dx,edge_order=1)[0], np.gradient(vy, dx,edge_order=1)[1], np.gradient(vz, dx, edge_order=1)[2] 
        #dvxdx = set_periodic(dvxdx, vx)/dx
        #dvydy = set_periodic(dvydy, vy)/dx
        #dvzdz = set_periodic(dvzdz, vz)/dx
        return dvxdx+dvydy+dvzdz     
    
    
    
    
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
    level= 9
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
        ds = an.analyse(outpath="/home/mattia/", outnumb=71)

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
    try:
        # load results, if presnt 
        c
        shocksfinder.load_results()
        
        
        
    except Exception as e:
        print(e)
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
    finally:
        # make a 3D plot with all the shocks found
        #shocksfinder.plot3D(types="sf", alpha=0.1, ss=1)
        # make histograms of the various shock quantities
        #shocksfinder.histograms(9, bins=50, log=True, alpha=0.5)
        #print("done")
        shocksfinder_results= shocksfinder.load_results()
        pass
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
    #shocksfinder.plot3D()
    #plt.show()
    quit()
