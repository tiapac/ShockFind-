import numpy as np 
import pickle as pk 
import pyvista as pv
import matplotlib.pyplot as plt
plt.style.use('dark_background')

plt.rcParams.update({
        'font.size'   : 18})
plt.rcParams['xtick.major.pad']='16'
plt.rcParams['ytick.major.pad']='16'
# initialize actor list 
level=256
class SetVisibilityCallback:
    """Helper callback to keep a reference to the actor being modified."""

    def __init__(self, actor, plotter,title, scalarbar):
        
        self.actor   = actor
        self.plotter = plotter 
        self.name    = title
        self.scalarbar=scalarbar
    def __call__(self, state):
        self.actor.SetVisibility(state)
        
        if self.scalarbar:
            if state: 
            
                self.plotter.add_scalar_bar(title  = self.name,
                                            color  = "white",
                                            mapper = self.actor.mapper)
            else:
                self.plotter.remove_scalar_bar(title=self.name)
                
            
def set_button(plotter,
               actor,
               title,
               scalarbar,
               color    = "white",
               inistate = True,
               posy     = 5.0,
               posx     = 5.0):
    callback = SetVisibilityCallback(actor,plotter,title,scalarbar)
    #if not inistate: 
    actor.SetVisibility(inistate)
    plotter.add_checkbox_button_widget(
        callback,
        value            =  inistate,
        position         =  (posx, posy),
        size             =  size,
        border_size      =  1,
        color_on         =  color,
        color_off        =  'grey',
        background_color =  'grey',
    )
    return

def add_vectors(plotter, sel_cond, func = np.log10, scale = 2, posy = 0.5, tol = 0.005, box=0.0,inistate_vec=False):
        new_cond = np.logical_and(cond, sel_cond)
        nx,ny,nz = shocks[0+3][new_cond],shocks[1+3][new_cond], shocks[2+3][new_cond]
        vshocks  = shocks[7][new_cond]
        x1,y1,z1 = shocks[0][new_cond], shocks[1][new_cond], shocks[2][new_cond]
        dx = box/level
        gridcut  = pv.StructuredGrid(x1*dx-box/2., y1*dx-box/2., z1*dx-box/2.,)
        vectors  = np.empty((gridcut.n_points, 3))
        # normalise vectors data to have values close to 1
        
        vectors[:, 0] = nx.flatten()#/vnorm
        vectors[:, 1] = ny.flatten()#/vnorm
        vectors[:, 2] = nz.flatten()#/vnorm
        
        title="V_sh [ufunc(cm/s)]"#r"$v_{\rm sh}$ ufunc [$\rm{cm\,s^{-1}}$] "
        #print("Vectors done.")
        gridcut["vectors"]  = vectors
        gridcut[title]      = func(vshocks.flatten())/scale
        gridcut.set_active_vectors("vectors")
        arrows=gridcut.glyph(orient   = True,
                            tolerance = tol,
                            scale     = title)
        actor = plotter.add_mesh(arrows, opacity = 0.5,
                        scalar_bar_args          = {"color":"white"},
                        show_scalar_bar          = False)       
        #actor.GetProperty().getScaleFactor(1)
        set_button(plotter,actor,title,scalarbar = True, 
                                        color    = "green",
                                        posx     = 20 + size // 2,
                                        posy     = posy,
                                        inistate = inistate_vec
                                        )

    
def add_a_conditioned_grid(sel_cond         = True,
                            qidx            = 6,
                            title           = None,
                            color           = "green",
                            posy            = 0.5,
                            unicolor        = None,
                            inistate        = True,
                            add_vector_field= True,
                            ufunc           = lambda a:a,
                            cmap            = "jet",
                            scale           = 1,
                            tol             = 0.005,
                            inistate_vec=False,
                            box=0.0,
                            ):
    if title is None: title = "No name"
    new_cond           = np.logical_and(cond, sel_cond)
    x1,y1,z1           = shocks[0][new_cond],shocks[1][new_cond],shocks[2][new_cond]
    family             = shocks[qidx][new_cond]
    dx = box/level
    points             = np.column_stack((x1*dx-box/2., y1*dx-box/2., z1*dx-box/2.))       
    point_cloud        = pv.PolyData(points)
    point_cloud[title] = ufunc(family)/scale
    # Create a PyVista plotter

    print("Building plot for index %s... "%title, end="")
    
    
    if unicolor is not None: 
        scalar_name = None
        cmap        = None
    else: 
        scalar_name = title
    actor     = plotter.add_points(
                    points          = point_cloud,
                    scalars         = scalar_name,
                    point_size      = pointsize,#0.8,
                    color           = unicolor,
                    cmap            = cmap,
                    style           = "points_gaussian",
                    emissive        = True,
                    log_scale       = True,
                    diffuse         = diffusivity,
                    nan_opacity     = 0.,
                    show_scalar_bar = False
                    )
    
    if unicolor is None and inistate==True: plotter.add_scalar_bar(title  = title,
                                                                   color  = "white",
                                                                   mapper = actor.mapper)
    
    set_button(plotter,actor,title, scalarbar = unicolor is None, inistate = inistate, color = color, posy = posy)
    
    plotter.add_text(
                title,
                position  = (40 + size//2 ,posy),
                color     = 'white',
                shadow    = False,
                font_size = 8,
            )
    if add_vector_field: add_vectors(plotter,
                                     new_cond,
                                     func  = np.log10,
                                     scale = 2,
                                     posy  = posy,
                                     tol   = tol, 
                                     box=box,
                                     inistate_vec=inistate_vec)
    
    actor_list.append(actor)
    print("Done.")
    return
    
    
class apply_allD:
    def __init__(self, actors,plotter):#,**keywargs):
        self.output  = actors  # Expected PyVista mesh type
        self.routines= []
        for actor in actors:
            self.routines.append(DiffuseSlider(actor = actor, plotter = plotter))
        self.plotter = plotter
    def __call__(self, value):
        for run in self.routines:
            run(value)
            self.update()
        return    
    def update(self):
        # This is where you call your simulation
        self.plotter.render()
        return

class DiffuseSlider:
    def __init__(self, actor,plotter):#,**keywargs):
        self.output  = actor  # Expected PyVista mesh type
        self.plotter = plotter
    
    def __call__(self, value):
        self.output.GetProperty().SetDiffuse(value)
        #self.update()
        return

    def update(self):
        # This is where you call your simulation
        self.plotter.render()
        return  
      
class apply_allS:
    """Not working 
    """
    def __init__(self, actors,plotter):#,**keywargs):
        self.output  = actors  # Expected PyVista mesh type
        self.routines= []
        for actor in actors:
            self.routines.append(PointSizeSlider(actor = actor, plotter = plotter))
        self.plotter = plotter
    def __call__(self, value):
        for run in self.routines:
            run(value)
            self.update()
        return    
    def update(self):
        # This is where you call your simulation
        self.plotter.render()
        return    
class PointSizeSlider:
    """not working. SetPointSize apply only if render_points_as_spheres is True
    """
    def __init__(self, actor,plotter):#,**keywargs):
        self.output  = actor  # Expected PyVista mesh type
        self.plotter = plotter
    
    def __call__(self, value):
        #print("hhh")
        self.output.GetProperty().SetPointSize(value)
        #self.update()
        return

    def update(self):
        # This is where you call your simulation
        self.plotter.render()
        return   
if __name__ == "__main__":
    #global actor_list  
    actor_list= []
    size        = 20  # button size
    pointsize   = 1.8 # base size of the points in the plot. The are then "splattered" in a gaussian profile
      # list of actors used to apply general changes to the scene
                      # updated at the creation of each actor
    diffusivity = 0.07 # set how "luminous the points are. "
    #with open("shocks+header_lvl9LB.pk", 'rb') as handle:
#    with open("pickled/shocks+header.pk", 'rb') as handle:
    with open("SNR_MHB_shocks+header_8.pk", 'rb') as handle:
                shocksfinder_results=  pk.load(handle)
    shocks = shocksfinder_results[0]
    header = shocksfinder_results[1]        
                                           # both types   # no error flags     # only conv. peaked at the center
    cond   = np.logical_and(np.logical_and( shocks[6] > 0,    shocks[15]==0),      shocks[14]==1)
    condF  = np.logical_and(np.logical_and( shocks[6] == 12,    shocks[15]==0),    shocks[14]==1)
    condS   = np.logical_and(np.logical_and( shocks[6] == 34,    shocks[15]==0),    shocks[14]==1)
    print(header)
    ind=7
    ss = shocks
    
    fig=plt.figure(2)
    ax=fig.add_subplot(111)
    bbins=50
    alpha=0.8
    #alpha1=1
    #ax.hist(ss[ind],  log=False, color="midnightblue", bins=bbins, alpha=alpha, label="Fast n = %5d"%len(ss[ind][cond]))
    ax.hist(ss[ind][condS]/1e5,log=True, color="red",bins=bbins, alpha=alpha,    label="Slow n = %5d"%len(ss[ind][condS]))
    ax.hist(ss[ind][condF]/1e5,log=True, color="blue", bins=bbins, alpha=alpha,  label="Fast n = %5d"%len(ss[ind][condF]))
    ax.set_xlabel(r"$v_{\rm sh }\,[{\rm kms^{-1}}]$")
    plt.tight_layout()
    #ax.legend()
    fig.savefig("shock_velocty_MHB")
    fig1=plt.figure(3)
    ind = 9
    ax1=fig1.add_subplot(111)
    ax1.hist(ss[ind][condS],density=True,log=True, color="red",bins=bbins, alpha=alpha,    label="Slow n = %5d"%len(ss[ind][condS]))
    ax1.hist(ss[ind][condF],density=True,log=True, color="blue", bins=bbins, alpha=alpha,  label="Fast n = %5d"%len(ss[ind][condF]))
    ax1.set_xlabel(r"$M_A$")
    plt.tight_layout()
    ax1.legend()
    fig1.savefig("shock_mach_MHB")
    
    
    ##ax.set_xlabel(r"$v_{sh}\,\,[km\,s^{-1}]$")
    ##ax.set_ylabel(r"$N_\{rm cells}$")
    ##ax.set_xlim(ss[ind].min(), ss[ind].max())
    plt.show()