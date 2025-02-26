import analysis_playground.analysis_tools as an
import numpy as np
import matplotlib.pyplot as plt
#import scipy
#import matplotlib.colors as mcolors
import shockfindCore.shockfind
import yt
import pickle
from mpl_toolkits.axes_grid1 import AxesGrid
import sys

class shocks:
    def __init__(self, outpath="./", outnumb=0, mhd=True,namefile="./pickled_stuff", lvl=None, bounds=None):
        self.out=outnumb
        self.outpath=outpath
        self.mhd=mhd
        self.namefile=namefile
        self.ds=ds=an.analyse(outpath=self.outpath, outnumb=self.out, select_data=bounds)
        self.ds.init_sp()
        self.data_ready=False


        self.mydiv      = None
        self.grad       = None
        self.grad_comps = None
        self.rho        = None
        self.pres       = None
        self.field      = None
        self.vel        = None
        self.vmean      = None
        #self.rhomean=None
        self.sigma      = None
        #self.dx=None
        self.lvl        = lvl

        #self.conv_thres=None
        #self.grho_thres=None
        

    def get_data_shockfind(self, load_data=True, save_data=True):
        if not self.data_ready:
            try: 
                if not load_data: raise(Exception)
                a=self.load_data()
                self.mydiv, self.grad,self.grad_comps, self.rho, self.pres, self.field, self.vel, self.vmean, self.rhomean, self.sigma, self.dx, prelvl= a
                if prelvl!=self.lvl:
                    self.mydiv, self.grad,self.grad_comps, self.rho, self.pres, self.field, self.vel, self.vmean, self.rhomean, self.sigma, self.dx=\
                    None, None, None, None, None, None, None, None, None, None, None
                    raise(Exception, "Wrong level requested.")
                print("Data found at %s loaded. " %(self.namefile+".pickle"))
                self.data_ready=True
            except:

                print("No data found at %s. Computing. " %(self.namefile+".pickle"))
                ds      = self.ds
                sigma   = (ds.std("velocity_magnitude").in_cgs()).d
                rhomean = (ds.avg("density").in_cgs()).d
                vmean   = (ds.avg("velocity_magnitude").in_cgs()).d
                print(self.lvl, "level")
                mydiv, cube=ds.get_cube(self.lvl, "mydiv", ghost=True)
                grad=cube["density_gradient_magnitude"]
                grad_comps=field=(
                                (cube["density_gradient_x"].to("g/cm**4")).d,
                                (cube["density_gradient_y"].to("g/cm**4")).d,
                                (cube["density_gradient_z"].to("g/cm**4")).d)

                rho=(cube["density"].to("g/cm**3")).d
                pres=(cube["pressure"].to("dyne/cm**2")).d
                field=((cube["magnetic_field_x"].to("Gauss")).d,
                    (cube["magnetic_field_y"].to("Gauss")).d,
                    (cube["magnetic_field_z"].to("Gauss")).d)
                vel=((cube["velocity_x"].to("cm/s")).d,
                    (cube["velocity_y"].to("cm/s")).d,
                    (cube["velocity_z"].to("cm/s")).d)
                mydiv=mydiv.to("1/s").d
                grad=grad.to("g/cm**4").d
                if self.lvl<ds.base_level: 
                    print("The coarser level is l=%d, setting variable level=%d"%(ds.base_level,ds.base_level))
                    self.lvl=ds.base_level
                elif self.lvl>ds.base_level+ds.maxlevel:
                    self.lvl=ds.base_level+ds.maxlevel
                    print("The maximum level is l=%d, setting variable level=%d"%((ds.base_level+ds.maxlevel),(ds.base_level+ds.maxlevel)))

                dx=(ds.width/2**(self.lvl)).to("cm").d[0]
                a=mydiv, grad,grad_comps, rho, pres, field, vel, vmean, rhomean, sigma, dx, self.lvl
                self.mydiv, self.grad,self.grad_comps, self.rho, self.pres, self.field, self.vel, self.vmean, self.rhomean, self.sigma, self.dx, self.lvl=a
                if save_data: self.save_data(a)
            
                self.data_ready=True
        elif self.data_ready:
            pass
        else: 
            raise(Exception)
        vtot=np.sqrt(self.vel[0]**2+self.vel[1]**2+self.vel[2]**2)
        #print(vtot.max()/1e5)
        #print((self.vel[0]).max()/1e5,(self.vel[1]).max()/1e5,(self.vel[2]).max()/1e5, "massimi")
        return self.mydiv, self.grad, self.grad_comps, self.rho, self.pres, self.field, self.vel, self.vmean, self.rhomean, self.sigma, self.dx, self.lvl
    
    def candidates_parameters(self,N=3,cs="auto", ca="auto", rhomean="auto", Tlim=1e4 , compress=4):
        
        
        if rhomean=="auto":
            
            try:
                rhomean=self.rhomean
                print("rhomean was ready rhomean = ",self.rhomean)
            except: 
                print("Computing average density of the hot gas.")
                rhomean=self.ds.sp[("gas","density")]
                T=self.ds.sp[("gas","temperature")]
                rhomean=(rhomean[T>=Tlim]).mean()
                self.rhomean=rhomean.d
        else:
            self.rhomean=rhomean



        try:
            if cs=="auto" or ca=="auto":
                print("Looking for ICs.")
                ds=an.analyse(outpath=self.outpath, outnumb=1)
                ds.init_sp()
                self.cs=(ds.sp["sound_speed"].to("cm/s").mean()).d if cs=="auto" else cs
                print("C_S computed on the ICs." if cs=="auto" else "C_S saved.")
                
                self.ca=(ds.sp["alfven_speed"].to("cm/s").mean()).d if ca=="auto" else ca
                print("C_A computed on the ICs." if cs=="auto" else "C_A saved.")

            else:
                self.cs= cs
                self.ca= ca
                print("C_S and C_A saved.")
            fail=False
        except Exception as e:
            print("Tried to extract C_S and C_A from ICs. Failed. Error",e)
            fail=True
            if cs=="auto":
                try:
                    cs=self.cs
                    print("cs was ready cs = ",self.cs)
                except: 
                    print("Computing average sound speed of the hot gas.")
                    cs=self.ds.sp[("gas","sound_speed")]
                    T=self.ds.sp[("gas","temperature")]
                    cs=(cs[T>=Tlim]).mean()
                    self.cs=cs.d
            else:
                self.cs=cs
            if ca=="auto":
                try:
                    ca=self.ca
                    print("ca was ready ca = ",self.ca)
                except: 
                    print("Computing average alfven speed of the hot gas.")
                    ca=self.ds.sp[("gas","alfven_speed")]
                    T=self.ds.sp[("gas","temperature")]
                    ca=(ca[T>=Tlim]).mean()
                    self.ca=ca.d
            else:
                self.ca=ca
        uni="cm/s"
        print("found ca=%.4f %s, cs=%.4f %s"%(self.ca,uni,self.cs,uni))

        vshock=1.05*min([self.cs,self.ca])
        conv_thres=(vshock*(compress-1)/(compress*N))/(self.dx)#*yt.units.cm)
        grho_thres=(self.rhomean)*(compress-1)/(N*self.dx)

        self.conv_thres=conv_thres#.d
        self.grho_thres=grho_thres
        print(self.conv_thres)
        return self.conv_thres, self.grho_thres

    def find_shokslist(self, ):
        #print(-self.mydiv, self.grad,self.conv_thres, self.grho_thres)
        #plt.imshow(self.grad[:,:,64])
        #plt.show()
        #quit()
        shocks=shockfind.shockfind.sfind_shock_simple(-self.mydiv, self.grad,self.conv_thres, self.grho_thres)
        
        #shocks=shockfind.shockfind.find_shocks(-self.mydiv, self.grad,self.conv_thres, self.grho_thres, Ncells=1e6)
        shockslist=[]
        x,y,z=shocks[0]
        for i,_ in enumerate(shocks[0][0]):
            shockslist.append((x[i],y[i],z[i]))
        #if len(shockslist)==0: sys.exit("no shocklist")
        #print(len(shockslist))
        #input()
        self.shockslist=shockslist
        return shockslist    
    def extra_params(self,
                        method_norm="point_gradient",
                        periodic=[True,True,True],
                        Rgrad=3,#*,
                        Rcylinder=3,#*,
                        gamma=5./3.,
                        line_range=10,#,
                        method_plane="point_field",
                        field_ref=0,
                        shock_ratio=1.1,
                        offset=[0,0,0]  ):
        
        extra={"method_norm":method_norm,
        "periodic":periodic,
        "Rgrad":Rgrad,#*dx,
        "Rcylinder":Rcylinder,#*dx,
        "gamma":gamma,
        "line_range":line_range,#dx,
        "method_plane":method_plane,
        "field_ref":field_ref,
        "shock_ratio":shock_ratio,
        "offset":offset
       }
        self.extra=extra
        return extra
    



    def run(self,cs="auto", ca="auto",rhomean="auto", load_data=True, save_data=True,
            load_data_cube=True, 
            save_data_cube=True, 
            parallel=False, 
            ncpus=None,
            full=True,
            quiet=False,
            Tlim=1e4):
        if parallel: 
            charact_funtion=shockfind.shockfind.characterise_shocks_parallel
        else:
            charact_funtion=shockfind.shockfind.characterise_shocks
        try: 
            if not load_data: raise(Exception)
            results=self.load_data(add="_results")
            ss,header, extra, shockslist= results
            print("Data found at %s loaded. " %(self.namefile+"_results.pickle"))
        except Exception as e:
            mydiv, grad , grad_comps, rho, pres, field, vel, vmean, rhomean, sigma, dx,lvl, = \
                self.get_data_shockfind(load_data=load_data_cube, save_data=save_data_cube)
            
            if "self.conv_thres" in locals() and "self.grho_thres" in locals(): 
                pass
            else: 
                    self.candidates_parameters(cs=cs, ca=ca,rhomean=rhomean, Tlim=Tlim)
            #print(self.conv_thres, self.grho_thres)
            print(e, "error. Searching for shocks candidates.")    
            if "self.shockslist" not in locals(): self.shockslist=self.find_shokslist()
            shockslist=self.shockslist
            
            try:
                extra=self.extra
            except Exception as e :
                print("Defining default parameter in Extra")
                extra=self.extra_params()

            #print(shockslist)
            #plt.imshow(rho[:,:,64])
            #plt.show()
            #sys.exit()
            
            self.tot = charact_funtion(shockslist, rho, pres, field, vel, mydiv, grad_comps, extra, quiet=quiet, ID='')#, ncpus=ncpus)
            ss, header=self.tot
            
            
            results= ss,header, extra, self.shockslist
        
            
        if save_data: self.save_data(results, add="_results")


        computed_shocks=self.extract_shocks(ss)
        self.computed_shocks=computed_shocks
        return [computed_shocks, results] if full else computed_shocks
    
    def extract_shocks(self, ss):

        condF   = np.logical_and(ss[6]==12, ss[15]==0  )
        condS   = np.logical_and(ss[6]==34, ss[15]==0  )#True )#
        condboh = np.logical_and(ss[6]==0 , True )

        xsF,ysF,zsF= ss[0][condF],ss[1][condF],ss[2][condF]
        xsS,ysS,zsS= ss[0][condS],ss[1][condS],ss[2][condS]
        xsboh,ysboh,zsboh= ss[0][condboh],ss[1][condboh],ss[2][condboh]
        ##vectors
        nxsS,nysS,nzsS= ss[3][condS],ss[4][condS],ss[5][condS]
        nxsF,nysF,nzsF= ss[3][condF],ss[4][condF],ss[5][condF]
        ###
        if "self.dx" not in locals():
            self.dx=self.get_data_shockfind(load_data=True, save_data=True)[10]
        
        
        Fshocks     = [xsF*self.dx,ysF*self.dx,zsF*self.dx]
        Sshocks     = [xsS*self.dx,ysS*self.dx,zsS*self.dx]
        Bohshocks   = [ xsboh*self.dx,ysboh*self.dx,zsboh*self.dx]
        ###
        Fpointers=[nxsS,nysS,nzsS]
        Spointers=[nxsF,nysF,nzsF]
        lists=[Fshocks, Sshocks, Bohshocks, Fpointers, Spointers]
        return lists
    def plot_candidates(self, ax=None, fig=None, alpha=0.5, ss=2):
        if fig==None: fig=plt.figure(figsize=(8,8))
        if ax==None: ax=fig.add_subplot(1, 1, 1, projection = '3d')
        xs,ys,zs=[],[],[]
        for s in self.shockslist:
            
            xs.append(s[0])
            ys.append(s[1])
            zs.append(s[2])
        p =ax.scatter(xs,ys,zs, alpha=alpha, s=ss, color="midnightblue" )
        return ax, fig 
    
    def plot2D(self,quantity="density", axis="z", log=True):
        s=self.ds.Splot(quantity, axis=axis, cmap="plasma", log=True)
        #s.set_zlim("mach_slow", 0, 10)
        s.set_log(quantity, log)
        s.annotate_magnetic_field()
        s.display()

        return s
    
    def calc_slices(self, scale=1, axis="z", lim=1):
        print(axis)
        if "self.dx" not in locals():
            self.dx=self.get_data_shockfind(load_data=True, save_data=True)[10]
        
        fac=1#self.dx/scale
        move  = -self.dx*2**(self.lvl-1) + 0.5*self.dx
        print(self.dx*2**(self.lvl-1)/scale,0.5*self.dx/scale )
        move /=  scale
        Fshocks,Sshocks,Bohshocks,Fpointers,Spointers=self.computed_shocks
        xsF,ysF,zsF=Fshocks
        xsS,ysS,zsS=Sshocks
        xsboh,ysboh,zsboh = Bohshocks
        #print( xsF/scale,self.dx/scale, scale, move )

        xsF  =  xsF/scale + move #*fac+move
        ysF  =  ysF/scale + move #*fac+move
        zsF  =  zsF/scale + move #*fac+move
        xsS  =  xsS/scale + move #*fac+move
        ysS  =  ysS/scale + move #*fac+move
        zsS  =  zsS/scale + move #*fac+move
        xsboh=xsboh/scale + move #*fac+move
        ysboh=ysboh/scale + move #*fac+move
        zsboh=zsboh/scale + move #*fac+move
        

        
        if axis=="z":
            plane=zsF
        elif axis=="y":
            plane=ysF
        elif axis=="x":
            plane=xsF
        pcond=np.logical_and(plane>-lim, plane<lim)
        xsliceF, ysliceF, zsliceF= xsF[pcond], ysF[pcond] ,zsF[pcond]

        if axis=="z":
            plane=zsS
        elif axis=="y":
            plane=ysS
        elif axis=="x":
            plane=xsS

        pcond=np.logical_and(plane>-lim, plane<lim)
        xsliceS, ysliceS, zsliceS= xsS[pcond], ysS[pcond] ,zsS[pcond]
        
        if axis=="z":
            return xsliceF, ysliceF, xsliceS, ysliceS
        elif axis=="y":
            return xsliceF, zsliceF, xsliceS, zsliceS
        elif axis=="x":
            return ysliceF, zsliceF, ysliceS, zsliceS
        

    def plot2DD(self,quantity="density", axis="z",scale=1, lim=1, units=None, log=True, fig=None):
        if fig is None: fig=plt.figure(figsize=(8,8))
    

        ax = self.ds.Splotax(quant=quantity, fig=fig,axis=axis, stream_m=True)
        #ax=plt.subplot(111)
        xsliceF,ysliceF,xsliceS,ysliceS =self.calc_slices( scale, axis, lim)
        #print( xsliceS,xsliceF)
        ax[0].scatter(xsliceF,ysliceF, color="midnightblue", s=1, alpha=1 )
        ax[0].scatter(xsliceS,ysliceS, color="red", s=1, alpha=1 )
        #print(xsliceS, xsliceF)
        #ax.scatter(xsS,ysS, alpha=0.5, s=0.5, color="red" )
        #set_axes_equal(ax)
        #
        #ax.set_xlabel("x")
        #ax.set_ylabel("y")
        #ax.set_zlabel("z")
        #plt.show()
        return ax, fig

    def plot3D(self, ax=None, fig=None, alpha=0.5, ss=2, types="fs"):
        types=types.upper()
        if fig==None: fig=plt.figure(figsize=(8,8))
        if ax==None: ax=fig.add_subplot(1, 1, 1, projection = '3d')
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
        #ax.scatter(xsS,ysS, alpha=0.5, s=0.5, color="red" )
        self.set_axes_equal(ax)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        #plt.show()

        return ax, fig
    


    def save_data(self, datas, add=""):
        with open(self.namefile+add+".pickle", 'wb') as handle:
          pickle.dump(datas, handle)
        return


    def load_data(self, add=""):

        with open(self.namefile+add+".pickle", 'rb') as handle:
            a=pickle.load( handle)
        return a
    


    def set_axes_equal(self, ax):
        """
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        """

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
