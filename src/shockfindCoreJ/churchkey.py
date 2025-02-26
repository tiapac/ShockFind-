#Copyright 2016 Andrew Lehmann
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import h5py
from numpy import array as nparray

#######################################################################
def churchkey(name, params):
    '''
    This function opens and returns a simulation datacube. It is given
    the name of the desired datacube, and returns a numpy array of that 
    datacubes, no matter the original structure.
    
    Parameters
    ----------
    name : str
        name of desired datacubes; must be able to take
            'rho', 'bx', 'by', 'bz', 'vx', 'vy', 'vz', 'pres', 'gradx', 'grady', 'gradz', 'divv'
        
    params: dict
        dictionary of config file entries. for this function, we need
        the simulation datacube locations, which can be accessed as
         params['file_rho'] for the density cube
         params['file_pres'] for the pressure cube
         params['file_bx'] for the bx cube
         params['file_by'] for the by cube
         params['file_bz'] for the bz cube
         params['file_vx'] for the vx cube
         params['file_vy'] for the vy cube
         params['file_vz'] for the vz cube
         params['file_div'] for the divergence cube
         params['file_gradx'] for the x-gradient cube
         params['file_grady'] for the y-gradient cube
         params['file_gradz'] for the z-gradient cube
        and we need the open method
         params['open_method']
        and if the following is false
         params['blind']
        then we need
         params['x0'] for lower x-limit to open
         params['y0'] for lower y-limit to open
         params['z0'] for lower z-limit to open
         params['xmax'] for upper x-limit to open
         params['ymax'] for upper y-limit to open
         params['zmax'] for upper z-limit to open
        
    
    Returns
    -------
    3D numpy array
        numpy array of desired cube         
    '''
    
    if params['open_method'] == 'federrath_hdf5':
        '''
        Doc string for Federrath cubes
        
        '''
        path = params['file_{0:s}'.format(name)]
        
        if name == 'rho':
            key = 'dens'
        elif name == 'pres':
            key = 'pres'
        elif name == 'bx':
            key = 'magx'
        elif name == 'by':
            key = 'magy'
        elif name == 'bz':
            key = 'magz'
        elif name == 'vx':
            key = 'velx'
        elif name == 'vy':
            key = 'vely'
        elif name == 'vz':
            key = 'velz'
        elif name == 'div':
            key = 'div'
        elif name == 'gradx':
            key = 'gradx'
        elif name == 'grady':
            key = 'grady'
        elif name == 'gradz':
            key = 'gradz'
            
    
        f = h5py.File(path, 'r')
        if params['blind']:
            cube = nparray(f[key][:])        
        else:
            cube = nparray(f[key][params['x0']:params['xmax'], params['y0']:params['ymax'], params['z0']:params['zmax']])
        f.close()
        
        return cube


    elif params['open_method'] == 'user_example':
        '''
        Here's your chance to add your own opening function.
        Study the above example!
        '''
        print 'user_example doesn\'t exist yet.'
        quit()
        
    else:
        print '*** The opening method \'{0:s}\' doesn\'t exist.'
        print '*** Check the churchkey.py script (and docs, if I\'ve made that).'
        quit()
#######################################################################
