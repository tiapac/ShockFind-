import numpy as np 
import pickle as pk 
import pyvista as pv

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
    with open("SNR_MHB_shocks+header_9.pk", 'rb') as handle:
                shocksfinder_results=  pk.load(handle)
    shocks = shocksfinder_results[0]
    header = shocksfinder_results[1]        
                                           # both types   # no error flags     # only conv. peaked at the center
    cond   = np.logical_and(np.logical_and( shocks[6] > 0,    shocks[15]==0),    1) #shocks[14]==1)
    

    first=True
    plotter = pv.Plotter(off_screen=False)
    NR_FRAMES = 1
    box=150
    dtheta = 360/NR_FRAMES
    poss=np.array((plotter.generate_orbital_path(factor = 0.9*box, n_points=NR_FRAMES+1, viewup=(0,0,1), shift=box/5)).points, dtype = float)
    plotter.camera.position=poss[0]
    for ii,pos in enumerate(poss):
#        print(ii)
#        add_a_conditioned_grid(sel_cond = shocks[6]==12, title = "FS", color = "blue",posy = size*2+size//10, unicolor = "blue" ,inistate = True, add_vector_field=True, box=150)
#        if ii<120 or ii>280:
#            diffusivity= 0.05
        add_a_conditioned_grid(sel_cond = shocks[6]==34, title = "", color = "red", posy = size*3+size//10, unicolor = "red"  ,
                                   inistate = True, add_vector_field=True, box=150, inistate_vec=ii>280)
#        else:
        diffusivity = 1
        add_a_conditioned_grid(sel_cond = shocks[6]==12, title = "", color = "blue",posy = 				size*2+size//10, unicolor = "blue" ,
                                   inistate = True, add_vector_field=True, box=150, inistate_vec=ii>160)
        
        
        if 1:
            for i in range(7,15):
                possss = i
                add_a_conditioned_grid(sel_cond = shocks[7] > 1e5, # only shocks moving at at least a km/s
                                    qidx     = i,
                                    title    = header[i],
                                    ufunc    = lambda a: a,
                                    color    = "yellow",
                                    posy     = size*possss + size//10,
                                    inistate = False ,
                                    add_vector_field = False,
                                    box=150)
                pos+=1

        if first:
            plotter.camera.is_set = True
            first                 = False
        plotter.set_background("black")
        if 0:plotter.show_bounds(    color     = "white",
                                bold      = False,
                                location  = 'origin',                       
                                ticks     = 'both',                       
                                n_xlabels = 4,                        
                                n_ylabels = 4,                        
                                n_zlabels = 4,                        
                                xtitle    = "x [pc]",                       
                                ytitle    = "y [pc]",                      
                                ztitle    = "z [pc]",    
                                font_size = 20,
                                )
        engineD = apply_allD(actor_list,plotter)
            #
        plotter.camera.azimuth+=dtheta#360/NR_FRAMES
        #plotter.camera.Elevation=90
        #plotter.camera.roll=0
        #plotter.distance=1000
        plotter.camera.focal_point=(0,0,0)
        if 1: plotter.add_slider_widget(
            callback     = lambda value: engineD(10**value),
            rng          = [-2, 0],
            value        = -1,
            title        = "Log10(Diffuse)",
            pointa       = (0.8, 0.8),
            pointb       = (1.0, 0.8),
            title_height = 0.03,
            title_color  = "white",
            color        = "white",
            slider_width = 0.01,
            tube_width   = 0.001
            
        )
        
        #engineS = apply_allS(actor_list,plotter)
        #
        #plotter.add_slider_widget(
        #    callback     = lambda value: engineS(value),
        #    rng          = [0, 100],
        #    value        = 0.2,
        #    title        = "Point size",
        #    pointa       = (0.8, 0.6),
        #    pointb       = (1.0, 0.6),
        #    title_height = 0.03,
        #    title_color  = "white",
        #    color        = "white",
        #    slider_width = 0.01,
        #    tube_width   = 0.001
        #    
        #)
        plotter.show(auto_close=False)
        
            
        plotter.screenshot("presentation/shocks_%05d"%ii)
        plotter.clear_actors()
