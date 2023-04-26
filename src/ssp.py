import numpy as np
import glob
import os
import copy
import scipy as scipy
import scipy.stats as stats
import scipy.linalg as linalg
import spharm
import sys

#================================================================================
def isotropise_topography(field, calc_mag_first=False, calc_avg=True):
    T,T = np.shape(field)
    if calc_mag_first:
        field_iso = np.array(np.zeros((T), dtype=np.float64))
    else:
        field_iso = np.array(np.zeros((T), dtype=np.complex64))
    for j in range(0,T):
        for i in range(0,j+1):
            sym_count=2.0
            if (i==0):
                sym_count=1.0
            if calc_mag_first:
                field_iso[j] += np.real( np.conj( field[i,j] ) * field[i,j] ) * sym_count
            else:
                field_iso[j] += field[i,j] * sym_count
        if calc_avg:
            field_iso[j] /= float(2.0*j+1.0)
    if calc_mag_first:
        return np.sqrt(field_iso)
    else:
        return field_iso

#================================================================================
def isotropise_field(field, calc_mag_first=False, calc_avg=True):
    nlev,T,T = np.shape(field)
    if calc_mag_first:
        field_iso = np.array(np.zeros((nlev,T), dtype=np.float64))
    else:
        field_iso = np.array(np.zeros((nlev,T), dtype=np.complex64))
    for j in range(0,T):
        for i in range(0,j+1):
            sym_count=2.0
            if (i==0):
                sym_count=1.0
            if calc_mag_first:
                field_iso[:,j] += np.real( np.conj( field[:,i,j] ) * field[:,i,j] ) * sym_count
            else:
                field_iso[:,j] += field[:,i,j] * sym_count
        if calc_avg:
            field_iso[:,j] /= float(2.0*j+1.0)
    if calc_mag_first:
        return np.sqrt(field_iso)
    else:
        return field_iso

#================================================================================
def isotropise_field_samples(field, calc_mag_first=False, calc_avg=True):
    nlev,T,T,num_samples = np.shape(field)
    if calc_mag_first:
        field_iso = np.array(np.zeros((nlev,T), dtype=np.float64))
    else:
        field_iso = np.array(np.zeros((nlev,T), dtype=np.complex64))
    for j in range(0,T):
        for i in range(0,j+1):
            sym_count=2.0
            if (i==0):
                sym_count=1.0
            if calc_mag_first:
                field_iso[:,j] += np.mean(np.real(np.abs(field[:,i,j,:]) * np.abs(field[:,i,j,:])),axis=1) * sym_count
            else:
                field_iso[:,j] += np.mean(field[:,i,j,:],axis=1) * sym_count
        if calc_avg:
            field_iso[:,j] /= float(2.0*j+1.0)
    if calc_mag_first:
        return np.sqrt(field_iso)
    else:
        return field_iso

#================================================================================
def isotropise_matrix(matrix):
    nlev,nlev,T,T = np.shape(matrix)
    matrix_iso = np.array(np.zeros((nlev,nlev,T), dtype=np.complex64))
    for j in range(0,T):
        for i in range(0,j+1):
            sym_count=2.0
            if (i==0):
                sym_count=1.0
            matrix_iso[:,:,j] += matrix[:,:,i,j] * sym_count
        matrix_iso[:,:,j] /= float(2.0*j+1.0)
    return matrix_iso

#================================================================================
def read_matrix_aniso(inputfilename):
    raw_data = read_ascii_2d(inputfilename)
    n = raw_data[0,:,0]
    m = raw_data[1,0,:]
    Tr_n = len(n)
    Tr_m = len(m)    
    num_var = int( (len(raw_data[:,0,0])-1)/2/2 )
    coef = np.array(np.zeros((num_var,num_var,Tr_n,Tr_m), dtype=np.complex64))
    for i in range(0,num_var):
        for j in range(0,num_var):        
            pos = (i*num_var+j)*2 + 2
            coef[i,j,:,:].real = raw_data[pos,:,:]
            coef[i,j,:,:].imag = raw_data[pos+1,:,:]        
    return coef, n, m

#================================================================================
def read_ascii_2d(inputfilename):
    f = open(inputfilename, 'r')
    N=0
    line_list = [0]
    while len(line_list)>0:
        line_list = f.readline().strip().split()
        if len(line_list)>0:
            if line_list[0] != '#':
                N=N+1
                Ncol = len(line_list)
    f.close()

    f = open(inputfilename, 'r')
    raw_data = np.array(np.zeros((Ncol, N, N), dtype=np.float64))
    j=0
    for line in f:
        line_list = line.strip().split()
        if len(line_list)>0:        
            if line_list[0] != '#':
                j = int(line_list[0])
                k = int(line_list[1])
                for i in range(0,Ncol):
                    raw_data[i,j,k] = float(line_list[i])
    f.close()
    return raw_data

#================================================================================
def read_topography(inputfilename):
    raw_data = read_ascii_2d(inputfilename)
    n = raw_data[0,:,0]
    m = raw_data[1,0,:]
    Tr_n = len(n)
    Tr_m = len(m)    
    num_lev = 2
    coef = np.array(np.zeros((Tr_n,Tr_m), dtype=np.complex64))
    coef[:,:].real = raw_data[2,:,:]
    coef[:,:].imag = raw_data[3,:,:]
    return coef, n, m

#================================================================================
def read_field_aniso(inputfilename_prefix):
    raw_data_L1 = read_ascii_2d(inputfilename_prefix + '.L01.dat')
    raw_data_L2 = read_ascii_2d(inputfilename_prefix + '.L02.dat')    
    n = raw_data_L1[0,:,0]
    m = raw_data_L1[1,0,:]
    Tr_n = len(n)
    Tr_m = len(m)    
    num_lev = 2
    coef = np.array(np.zeros((num_lev,Tr_n,Tr_m), dtype=np.complex64))
    coef[0,:,:].real = raw_data_L1[2,:,:]
    coef[0,:,:].imag = raw_data_L1[3,:,:]
    coef[1,:,:].real = raw_data_L2[2,:,:]
    coef[1,:,:].imag = raw_data_L2[3,:,:]    
    return coef, n, m

#================================================================================
def read_spectral_components(simulation_dir, timestep=7000):
    a     = 6.371e6   # radius of the Earth (m)
    
    Etot, n = read_spectra(simulation_dir+'/results/spectra/spectra.energy_k.n.'+'{:06d}'.format(timestep))
    vort_avg = read_field_aniso(simulation_dir+'/results/fields/vort.avg')[0]
    Eavg = isotropise_field((vort_avg*vort_avg.conj()).real/8.0, calc_avg=False)[:,1:]/n/(n+1.0)*a*a
    Eprime = Etot - Eavg
    
    vort_topo = read_topography(simulation_dir+'/results/initial_conditions/topography.scaled.dat')[0]
    Etopo_avg = copy.deepcopy(Eavg)
    Etopo_avg[0,:] = (isotropise_topography(np.abs(vort_avg[0,:,:]*vort_topo.conj()).real, calc_avg=False))[1:]/n/(n+1.0)*a*a
    Etopo_avg[1,:] = (isotropise_topography(np.abs(vort_avg[1,:,:]*vort_topo.conj()).real, calc_avg=False))[1:]/n/(n+1.0)*a*a
    
    Etopo = (isotropise_topography(np.abs(vort_topo*vort_topo.conj()), calc_avg=False)).real[1:]/n/(n+1.0)*a*a

    vort_nudge = read_field_aniso(simulation_dir + '/results/target_conditions/vort.target_climate')[0]
    Enudge = isotropise_field((vort_nudge*vort_nudge.conj()).real/8.0, calc_avg=False)[:,1:]/n/(n+1.0)*a*a;
    
    results = dict()
    results['Etot'] = Etot
    results['Eavg'] = Eavg
    results['Eprime'] = Eprime
    results['Etopo_avg'] = Etopo_avg
    results['Etopo'] = Etopo
    results['Enudge'] = Enudge
    results['n'] = n
    
    return results

#================================================================================
def read_spectra(inputfilename_prefix, trunc=False, read_min_max=False):
    if trunc:
        raw_data_L1 = read_ascii_1d(inputfilename_prefix + '.L01.trunc.dat')
        raw_data_L2 = read_ascii_1d(inputfilename_prefix + '.L02.trunc.dat')    
    else:
        raw_data_L1 = read_ascii_1d(inputfilename_prefix + '.L01.dat')
        raw_data_L2 = read_ascii_1d(inputfilename_prefix + '.L02.dat')

    k = raw_data_L1[0,:]
    num_lev = 2
    Tr = len(k)    
    coef = np.array(np.zeros((num_lev,Tr), dtype=np.float64))
    coef[0,:] = raw_data_L1[3,:]
    coef[0,:] = raw_data_L1[3,:]    
    coef[1,:] = raw_data_L2[3,:]
    coef[1,:] = raw_data_L2[3,:]    
    
    if read_min_max:
        coef_min = np.array(np.zeros((num_lev,Tr), dtype=np.float64))
        coef_min[0,:] = raw_data_L1[2,:]
        coef_min[0,:] = raw_data_L1[2,:]    
        coef_min[1,:] = raw_data_L2[2,:]
        coef_min[1,:] = raw_data_L2[2,:]    

        coef_max = np.array(np.zeros((num_lev,Tr), dtype=np.float64))
        coef_max[0,:] = raw_data_L1[4,:]
        coef_max[0,:] = raw_data_L1[4,:]    
        coef_max[1,:] = raw_data_L2[4,:]
        coef_max[1,:] = raw_data_L2[4,:]    

        return coef, k, coef_min, coef_max
    else:
        return coef, k

#================================================================================
def read_ascii_1d(inputfilename):
    f = open(inputfilename, 'r')
    N=0
    for line in f:
        line_list = line.strip().split()
        if line_list[0] != '#':
            N=N+1
    f.close()
    Ncol = len(line_list)

    f = open(inputfilename, 'r')
    raw_data = np.array(np.zeros((Ncol,N), dtype=np.float64))
    j=0
    for line in f:
        line_list = line.strip().split()
        if line_list[0] != '#':
            for i in range(0,Ncol):
                raw_data[i,j] = float(line_list[i])
            j=j+1
    f.close()
    return raw_data    

#================================================================================
def write_ascii_anisotropic_matrix(filename,data):
    nlev,nlev,T,T = np.shape(data)    
    file = open(filename,'w')
    for i in range(0,T):
        for k in range(0,T):
            file.write('%4.1d %4.1d %21.12e %21.12e %21.12e %21.12e %21.12e %21.12e %21.12e %21.12e\n' % (i, k, data.real[0,0,i,k], data.imag[0,0,i,k], data.real[0,1,i,k], data.imag[0,1,i,k], data.real[1,0,i,k], data.imag[1,0,i,k], data.real[1,1,i,k], data.imag[1,1,i,k]))
        file.write('\n')
    file.close()

#================================================================================
def write_ascii_isotropic_matrix(filename,data):
    nlev,nlev,T = np.shape(data)    
    file = open(filename,'w')
    for i in range(0,T):
        file.write('%4.1d %21.12e %21.12e %21.12e %21.12e %21.12e %21.12e %21.12e %21.12e\n' % (i, data.real[0,0,i], data.imag[0,0,i], data.real[0,1,i], data.imag[0,1,i], data.real[1,0,i], data.imag[1,0,i], data.real[1,1,i], data.imag[1,1,i] ))
    file.close()

#================================================================================
def write_ascii_anisotropic_vector(filename,data):
    nlev,T,T = np.shape(data)    
    file = open(filename,'w')
    for i in range(0,T):
        for k in range(0,T):
            file.write('%4.1d %4.1d %21.12e %21.12e %21.12e %21.12e\n' % (i, k, data.real[0,i,k], data.imag[0,i,k], data.real[1,i,k], data.imag[1,i,k]))
        file.write('\n')
    file.close()

#================================================================================
def write_ascii_isotropic_vector(filename,data):
    nlev,T = np.shape(data)    
    file = open(filename,'w')
    for i in range(0,T):
        file.write('%4.1d %21.12e %21.12e %21.12e %21.12e\n' % (i, data.real[0,i], data.imag[0,i], data.real[1,i], data.imag[1,i]))
    file.close()

#================================================================================
def write_ascii_field_2d(filename,data):
    nlev,T = np.shape(data)    
    file = open(filename,'w')
    for i in range(0,T):
        for k in range(0,T):
            file.write('%4.1d %4.1d %21.12e %21.12e\n' % (i, k, data.real[i,k], data.imag[i,k]))
        file.write('\n')
    file.close()
    return

#================================================================================
def convert_q_to_vort(FL,q_field):
    nlev,T,T = np.shape(q_field)
    vort_field = np.array(np.zeros((nlev,T,T), dtype=np.complex64))
    for j in range(0,T):
        for i in range(0,j+1):
            if j==0:
                denom = 1.0 + FL/2.0
            else:
                denom = 1.0 + FL/float(j)/(float(j)+1.0)
            BT = (q_field[0,i,j] + q_field[1,i,j])/2.0
            BC = (q_field[0,i,j] - q_field[1,i,j])/2.0/denom
            vort_field[0,i,j] =  BT + BC
            vort_field[1,i,j] =  BT - BC
    return vort_field

#================================================================================
def calculate_grid(T_R):
    nlat=int((T_R+1)/2*3)
    lat = spharm.gaussian_lats_wts(nlat)[0]
    lon = np.linspace(0,360,2*len(lat)+1)[0:-1]
    return lat, lon

#================================================================================
def convert_spect_2d_to_1d(spect_2d, T_R):
    spect_1d = np.array(np.zeros( ( int((T_R+2)*(T_R+1)/2) ),dtype=np.complex64))
    pos=0
    for m in range(0,T_R):
        for n in range(m,T_R+1):
            spect_1d[pos]=spect_2d[m,n]
            pos+=1
    return spect_1d

#================================================================================
def get_physical_fields(filename):
    T_recon=100
    vort_spectral_2d, n_dns, m_dns = read_field_aniso(filename)
    vort_spectral_2d = vort_spectral_2d[:,:T_recon,:T_recon]
    n_dns = n_dns[:T_recon]
    m_dns = m_dns[:T_recon]

    T_dns = len(n_dns)-1
    lat_dns, lon_dns = calculate_grid(T_dns)
    spharm_dns = spharm.Spharmt(nlat=len(lat_dns), nlon=len(lon_dns), gridtype='gaussian')

    vort_L1 = spharm_dns.spectogrd(convert_spect_2d_to_1d(vort_spectral_2d[0,:T_dns+1,:T_dns+1], T_dns))
    vort_L2 = spharm_dns.spectogrd(convert_spect_2d_to_1d(vort_spectral_2d[1,:T_dns+1,:T_dns+1], T_dns))

    strm = copy.deepcopy(vort_spectral_2d)
    a=6.371e6
    for ii in range(1,T_dns):
        laplacian = -1.0*ii*(ii+1)/a/a
        strm[:,:,ii] = strm[:,:,ii]/laplacian

    strm_L1 = spharm_dns.spectogrd(convert_spect_2d_to_1d(strm[0,:,:], T_dns))
    strm_L1_dx, strm_L1_dy = spharm_dns.getgrad(convert_spect_2d_to_1d(strm[0,:,:], T_dns))
    u_L1 = -strm_L1_dy; v_L1 = strm_L1_dx

    strm_L2 = spharm_dns.spectogrd(convert_spect_2d_to_1d(strm[1,:,:], T_dns))
    strm_L2_dx, strm_L2_dy = spharm_dns.getgrad(convert_spect_2d_to_1d(strm[1,:,:], T_dns))
    u_L2 = -strm_L2_dy; v_L2 = strm_L2_dx
    
    output = dict()
    output['lat'] = lat_dns
    output['lon'] = lon_dns
    output['vort L1'] = vort_L1
    output['vort L2'] = vort_L2
    output['strm L1'] = strm_L1
    output['strm L2'] = strm_L2
    output['u L1'] = u_L1
    output['u L2'] = u_L2
    output['v L1'] = v_L1
    output['v L2'] = v_L2
    
    output['vort BT'] = (vort_L1+vort_L2)/2
    output['vort BC'] = vort_L1-vort_L2
    output['strm BT'] = (strm_L1+strm_L2)/2
    output['strm BC'] = strm_L1-strm_L2
    output['u BT'] = (u_L1+u_L2)/2
    output['u BC'] = u_L1-u_L2
    output['v BT'] = (v_L1+v_L2)/2
    output['v BC'] = v_L1-v_L2
    
    return output

#================================================================================
class SubgridModel(object):

    #---------------------------------------------------------------------------
    def __init__(self, FL=4e4):
        self.omega   = 7.292e-5  # omega - rotational speed of Earth [1/s]
        self.FL      = FL     # layer coupling parameter
        self.read_meanfields = True     # flag to read mean climate sates
        return
    
    #---------------------------------------------------------------------------    
    def read_data(self, input_data_dir, spin_up=0, read_meanfields=True, num_meanfields=None, combined_dir='dns_combined'):
        self.input_data_dir = input_data_dir
        
        dns_dir_list = glob.glob(self.input_data_dir + '/dns??.????')
        
        topography_filename = dns_dir_list[0] + '/results/initial_conditions/topography.scaled.dat'
        if os.path.isfile(topography_filename):
            self.h = read_topography(topography_filename)[0] # units of metres
        else:
            self.h = None
            
        self.tauM = int(self.input_data_dir[-8:-4])
        print('Number of timesteps to calculate of each climate state = {0}'.format(self.tauM))

        print('Reading eddy-eddy model')        
        self.czero, self.n, self.m       =  read_matrix_aniso(input_data_dir + '/'+combined_dir+'/results/subgrid/czero_aniso.dat')     
        self.tzero =  read_matrix_aniso(input_data_dir + '/'+combined_dir+'/results/subgrid/tzero_aniso.dat')[0]
        self.cint  =  read_matrix_aniso(input_data_dir + '/'+combined_dir+'/results/subgrid/cint_aniso.dat')[0]
        self.tint  =  read_matrix_aniso(input_data_dir + '/'+combined_dir+'/results/subgrid/tint_aniso.dat')[0]
        
        self.Tr_n    = len(self.n)
        self.Tr_m    = len(self.m)
        self.num_lev = len(self.czero[:,0,0,0])
                
        print('Reading mean subgrid tendencies')
        self.f_subgrid_avg = np.array(np.zeros((self.num_lev,self.Tr_n,self.Tr_m), dtype=np.complex64))
        self.f_subgrid_avg = read_field_aniso(input_data_dir + '/'+combined_dir+'/results/subgrid/red_vort_subgrid_tend.avg_trunc')[0]/self.omega/self.omega
        self.f_subgrid_avg_iso = isotropise_field(self.f_subgrid_avg)
        self.f_subgrid_avg_mag_iso = isotropise_field(self.f_subgrid_avg, calc_mag_first=True)
                    
        self.read_meanfields = read_meanfields
        if self.read_meanfields:
            print('Reading climate states and tendencies')        
            self.dir_list = sorted(glob.glob(self.input_data_dir + '/dns??.????.meanfield_jacobian/*'))
            if num_meanfields is not None:
                self.dir_list = self.dir_list[0:num_meanfields]
                
            if len(self.dir_list)>spin_up:
                self.dir_list = self.dir_list[spin_up:]
            else:
                print('ERROR : spinup longer than record length.')
                sys.exit()
            self.num_samples = len(self.dir_list)
            print('   number of files = {0}'.format(self.num_samples))

            this_dir = self.dir_list[0]
            batch_name = this_dir[-36:-26]
            climate_name = this_dir[-6:]

            self.f = np.array(np.zeros((self.num_lev,self.Tr_n,self.Tr_m,self.num_samples), dtype=np.complex64))
            self.q = np.array(np.zeros((self.num_lev,self.Tr_n,self.Tr_m,self.num_samples), dtype=np.complex64))
            self.b = np.array(np.zeros((self.num_lev,self.Tr_n,self.Tr_m,self.num_samples), dtype=np.complex64))
            for i in range(0,self.num_samples):
                print('    sample {0} of {1}'.format(i+1,self.num_samples))
                this_dir = self.dir_list[i]
                batch_name = this_dir[-36:-26]
                climate_name = this_dir[-6:]
                f_filename = self.dir_list[i] + '/results/OLS/red_vort_subgrid_tend.avg_trunc.all_steps'
                q_filename = self.dir_list[i] + '/results/OLS/red_vort.avg_trunc.all_steps'
                b_filename = self.dir_list[i] + '/results/OLS/red_vort_subgrid_tend.avg_trunc'                
                self.f[:,:,:,i] = read_field_aniso(f_filename)[0]
                self.q[:,:,:,i] = read_field_aniso(q_filename)[0]
                self.b[:,:,:,i] = read_field_aniso(b_filename)[0]
            self.q = self.q/self.omega
            self.f = self.f/self.omega/self.omega
            self.b = self.b/self.omega/self.omega
            self.h = self.h/self.omega

            print('Calculating sample averages')
            self.f_avg = np.mean(self.f,axis=3)
            self.q_avg = np.mean(self.q,axis=3)
            self.b_avg = np.nanmean(self.b,axis=3)
            
            print('Isotropising fields and matrices')
            self.f_avg_iso       = isotropise_field(self.f_avg)
            self.q_avg_iso       = isotropise_field(self.q_avg)
            self.b_avg_iso       = isotropise_field(self.b_avg)
            self.f_avg_mag_iso   = isotropise_field(self.f_avg,  calc_mag_first=True)
            self.q_avg_mag_iso   = isotropise_field(self.q_avg,  calc_mag_first=True)
            self.b_avg_mag_iso   = isotropise_field(self.b_avg,  calc_mag_first=True)
            self.f_mag_iso   = isotropise_field_samples(self.f,  calc_mag_first=True)
            self.q_mag_iso   = isotropise_field_samples(self.q,  calc_mag_first=True)
            self.b_mag_iso   = isotropise_field_samples(self.b,  calc_mag_first=True)
        return
    
    #---------------------------------------------------------------------------    
    def calculate_eddy_coefficients(self, FL=4e4): 
        
        self.FL = FL
        
        print('Isotropising subgrid statistics')
        self.czero_iso = isotropise_matrix(self.czero)
        self.tzero_iso = isotropise_matrix(self.tzero)
        self.cint_iso  = isotropise_matrix(self.cint)
        self.tint_iso  = isotropise_matrix(self.tint)
        
        print('Calculating dissipation matrices and eigenvalues from anisotropic subgrid statistics')
        self.net         = copy.deepcopy(self.czero)*0.0
        self.drain       = copy.deepcopy(self.czero)*0.0
        self.backscatter = copy.deepcopy(self.czero)*0.0     
        self.bnoise      = copy.deepcopy(self.czero)*0.0
        self.evalue      = copy.deepcopy(self.czero[0,:,:,:])*0.0
        self.evector     = copy.deepcopy(self.czero)*0.0
        self.net_evalue  = copy.deepcopy(self.czero[0,:,:,:])*0.0
        self.drain_evalue= copy.deepcopy(self.czero[0,:,:,:])*0.0
        self.net_baro    = copy.deepcopy(self.czero)*0.0
        self.drain_baro  = copy.deepcopy(self.czero)*0.0 
        self.net_vort    = copy.deepcopy(self.czero)*0.0
        self.drain_vort  = copy.deepcopy(self.czero)*0.0 
        for ii in range(0,self.Tr_n):
            if ii%10 == 0:
                print('   {0} of {1}'.format(ii,self.Tr_n))
            for jj in range(ii,self.Tr_m):
                self.net[:,:,ii,jj], self.drain[:,:,ii,jj], self.backscatter[:,:,ii,jj], self.bnoise[:,:,ii,jj], self.evalue[:,ii,jj], self.evector[:,:,ii,jj], self.net_evalue[:,ii,jj], self.drain_evalue[:,ii,jj] \
                    = self.calculate_dissipation(ii, self.czero[:,:,ii,jj], self.tzero[:,:,ii,jj], self.cint[:,:,ii,jj], self.tint[:,:,ii,jj]) 
                self.net_baro[:,:,ii,jj]   = self.convert_evis_q_to_baro(ii,FL,self.net[:,:,ii,jj])
                self.drain_baro[:,:,ii,jj] = self.convert_evis_q_to_baro(ii,FL,self.drain[:,:,ii,jj])
                self.net_vort[:,:,ii,jj]   = self.convert_evis_baro_to_vort(self.net_baro[:,:,ii,jj])
                self.drain_vort[:,:,ii,jj] = self.convert_evis_baro_to_vort(self.drain_baro[:,:,ii,jj])
        self.backscatter_baro = self.net_baro - self.drain_baro
        self.backscatter_vort = self.net_vort - self.drain_vort
        
        print('   minimum drain eigenvalue = ',np.min(self.drain_evalue.real))
        print('   minimum net eigenvalue = ',np.min(self.net_evalue.real))
        
        print('Calculating dissipation matrices and eigenvalues from isotropic subgrid statistics')
        self.net_iso         = copy.deepcopy(self.czero_iso)*0.0
        self.drain_iso       = copy.deepcopy(self.czero_iso)*0.0        
        self.backscatter_iso = copy.deepcopy(self.czero_iso)*0.0        
        self.bnoise_iso      = copy.deepcopy(self.czero_iso)*0.0
        self.evalue_iso      = copy.deepcopy(self.czero_iso[0,:,:])*0.0
        self.evector_iso     = copy.deepcopy(self.czero_iso)*0.0
        self.net_evalue_iso  = copy.deepcopy(self.czero_iso[0,:,:])*0.0
        self.drain_evalue_iso= copy.deepcopy(self.czero_iso[0,:,:])*0.0
        for ii in range(0,self.Tr_n):
            self.net_iso[:,:,ii], self.drain_iso[:,:,ii], self.backscatter_iso[:,:,ii], self.bnoise_iso[:,:,ii], self.evalue_iso[:,ii], self.evector_iso[:,:,ii], self.net_evalue_iso[:,ii], self.drain_evalue_iso[:,ii] \
                = self.calculate_dissipation(ii, self.czero_iso[:,:,ii], self.tzero_iso[:,:,ii], self.cint_iso[:,:,ii], self.tint_iso[:,:,ii])
            
        print('Isotropising anisotropic dissipation matrices and calculating the eigenvalues')
        self.drain_iso_post       = isotropise_matrix(self.drain)
        self.backscatter_iso_post = isotropise_matrix(self.backscatter)
        self.net_iso_post         = isotropise_matrix(self.net)
        self.drain_evalue_iso_post= isotropise_field(self.drain_evalue)
        self.net_evalue_iso_post  = isotropise_field(self.net_evalue)
        self.bnoise_iso_post      = copy.deepcopy(self.czero_iso)*0.0
        self.evalue_iso_post      = copy.deepcopy(self.czero_iso[0,:,:])*0.0
        self.evector_iso_post     = copy.deepcopy(self.czero_iso)*0.0
        for ii in range(0,self.Tr_n):
            bnoise_tmp = -np.dot(self.backscatter_iso_post[:,:,ii], self.czero_iso[:,:,ii])
            self.bnoise_iso_post[:,:,ii] = bnoise_tmp + bnoise_tmp.T.conj()
            self.evalue_iso_post[:,ii], self.evector_iso_post[:,:,ii] = linalg.eig(self.bnoise_iso_post[:,:,ii])
        self.evalue_iso_post.real[self.evalue_iso_post.real<0.0] = 0.0
        self.evalue_iso_post.imag = 0.0

        return


    #---------------------------------------------------------------------------    
    def calculate_dissipation(self, ii, this_czero, this_tzero, this_cint, this_tint, tol=1.0e-15, cond_tol=100):
        net             = -np.dot(this_tzero, linalg.pinvh(this_czero, rcond=tol))            
        drain           = -np.dot(this_tint, linalg.pinvh(this_cint, rcond=tol))
        bnoise_temp     = -np.dot(net-drain, this_czero)
        if (np.linalg.cond(net)>cond_tol) or (np.linalg.cond(drain)>cond_tol) or np.isnan( np.sum(net).real + np.sum(drain).real + np.sum(bnoise_temp).real ):
            net[:,:] = 0.0
            drain[:,:] = 0.0
            bnoise_temp[:,:] = 0.0
        back     = net - drain    
        
        bnoise   = bnoise_temp + bnoise_temp.T.conj()
        evalue, evector = linalg.eig(bnoise)
        evalue.real[evalue.real<0.0] = 0.0
        evalue[:].imag = 0.0

        ev_net   = np.sort(linalg.eig(net)[0])
        ev_drain = np.sort(linalg.eig(drain)[0])

        return net, drain, back, bnoise, evalue, evector, ev_net, ev_drain
    

    #---------------------------------------------------------------------------    
    def calculate_meanfield_coefficients(self, tol=1e-15, cond_tol=100):
        if not self.read_meanfields:
            print('Meanfields not yet read from file.')
            return
        f           = self.f
        b           = self.b
        q           = self.q
        num_samples = len(f[0,0,0,:])

        print('Calculating meanfield coefficients')
        self.T0_meanfield = np.array(np.zeros((self.num_lev,self.num_lev,self.Tr_n,self.Tr_m), dtype=np.complex64))
        self.C0_meanfield = np.array(np.zeros((self.num_lev,self.num_lev,self.Tr_n,self.Tr_m), dtype=np.complex64))
        self.D_meanfield  = np.array(np.zeros((self.num_lev,self.num_lev,self.Tr_n,self.Tr_m), dtype=np.complex64))
        self.D_q          = np.array(np.zeros((self.num_lev,self.Tr_n,self.Tr_m,self.num_samples), dtype=np.complex64))
        A = np.array(np.zeros((self.num_lev,1), dtype=np.complex64))
        B = np.array(np.zeros((self.num_lev,1), dtype=np.complex64))                
        for ii in range(0,self.Tr_n):
            if ii%10 == 0:
                print('   {0} of {1}'.format(ii,self.Tr_n))
            for jj in range(ii,self.Tr_m):
                # Calculate covariances
                for kk in range(0,num_samples):
                    A[:,0] = f[:,ii,jj,kk] - self.f_avg[:,ii,jj] - b[:,ii,jj,kk] + self.b_avg[:,ii,jj]
                    B[:,0] = q[:,ii,jj,kk] - self.q_avg[:,ii,jj]
                    self.T0_meanfield[:,:,ii,jj] += np.matmul(A, B.conj().T)
                    self.C0_meanfield[:,:,ii,jj] += np.matmul(B, B.conj().T)                    
                # Calculate eddy-meanfield operator
                self.D_meanfield[:,:,ii,jj] = -np.dot(self.T0_meanfield[:,:,ii,jj],linalg.pinvh(self.C0_meanfield[:,:,ii,jj], rcond=tol))
                if (np.linalg.cond(self.D_meanfield[:,:,ii,jj])>cond_tol):
                    self.D_meanfield[:,:,ii,jj] = 0.0
                # Calculate eddy-meanfield force
                for kk in range(0,num_samples):
                    self.D_q[:,ii,jj,kk]    = np.matmul(self.D_meanfield[:,:,ii,jj], self.q[:,ii,jj,kk])
        self.T0_meanfield /= num_samples
        self.C0_meanfield /= num_samples
        
        # Calculate eddy-topographic terms                    
        self.D_q_avg   = np.mean(self.D_q,axis=3)
        self.chi_h_avg = self.f_avg - self.b_avg + self.D_q_avg
        self.chi_h     = self.f - self.b + self.D_q
        if self.h is not None:
            self.h_Tr         = self.h[0:self.Tr_n,0:self.Tr_m]            
            self.chi          = np.array(np.zeros((self.num_lev,self.Tr_n,self.Tr_m,self.num_samples), dtype=np.complex64))
            for ii in range(0,self.num_lev):
                for kk in range(0,num_samples):
                    self.chi[ii,:,:,kk] = self.chi_h[ii,:,:,kk] / self.h_Tr                
            self.chi[np.isnan(self.chi)] = 0.0
            self.chi_avg   = np.mean(self.chi,axis=3)            
        else:
            self.h_Tr    = None
            self.chi     = None            
            self.chi_avg = None
        
        print('Isotropising the fields and matrices')
        self.D_q_iso             = isotropise_field(self.D_q_avg)
        self.chi_h_iso           = isotropise_field(self.chi_h_avg)
        self.D_q_avg_mag_iso     = isotropise_field(self.D_q_avg,   calc_mag_first=True)        
        self.chi_h_avg_mag_iso   = isotropise_field(self.chi_h_avg, calc_mag_first=True)
        self.D_q_mag_iso         = isotropise_field_samples(self.D_q,   calc_mag_first=True)        
        self.chi_h_mag_iso       = isotropise_field_samples(self.chi_h, calc_mag_first=True)
        self.D_meanfield_iso     = isotropise_matrix(self.D_meanfield)
        if self.h is not None:
            self.chi_iso             = isotropise_field(self.chi_avg)
            self.chi_avg_mag_iso     = isotropise_field(self.chi_avg,   calc_mag_first=True)
            self.chi_mag_iso         = isotropise_field_samples(self.chi,   calc_mag_first=True)            
            self.h_iso               = isotropise_field(np.expand_dims(self.h_Tr, axis=0))
            self.h_mag_iso           = isotropise_field(np.expand_dims(self.h_Tr, axis=0),   calc_mag_first=True)     
        else:
            self.chi_iso             = None
            self.chi_avg_mag_iso     = None
            self.chi_mag_iso         = None
            self.h_iso               = None
            self.h_mag_iso           = None
                    
        return
    

    #---------------------------------------------------------------------------    
    def write_output_data(self, output_dir):
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(output_dir+'iso_pre'):
            os.makedirs(output_dir+'iso_pre')
        if not os.path.exists(output_dir+'iso_post'):
            os.makedirs(output_dir+'iso_post')
            
        scale_by = self.omega*self.omega
        
        write_ascii_anisotropic_matrix(output_dir + 'drain_aniso.dat',self.drain)
        write_ascii_anisotropic_matrix(output_dir + 'net_aniso.dat',self.net)
        write_ascii_anisotropic_matrix(output_dir + 'backscatter_aniso.dat',self.backscatter)
        write_ascii_anisotropic_matrix(output_dir + 'bnoise_aniso.dat',self.bnoise)
        write_ascii_anisotropic_matrix(output_dir + 'evector_aniso.dat',self.evector)
        write_ascii_anisotropic_vector(output_dir + 'evalue_aniso.dat',self.evalue)
        
        write_ascii_isotropic_matrix(output_dir + 'iso_pre/drain_iso.dat',self.drain_iso)
        write_ascii_isotropic_matrix(output_dir + 'iso_pre/net_iso.dat',self.net_iso)
        write_ascii_isotropic_matrix(output_dir + 'iso_pre/backscatter_iso.dat',self.backscatter_iso)
        write_ascii_isotropic_matrix(output_dir + 'iso_pre/bnoise_iso.dat',self.bnoise_iso)
        write_ascii_isotropic_matrix(output_dir + 'iso_pre/evector_iso.dat',self.evector_iso)
        write_ascii_isotropic_vector(output_dir + 'iso_pre/evalue_iso.dat',self.evalue_iso)
        
        write_ascii_isotropic_matrix(output_dir + 'iso_post/drain_iso.dat',self.drain_iso_post)
        write_ascii_isotropic_matrix(output_dir + 'iso_post/net_iso.dat',self.net_iso_post)
        write_ascii_isotropic_matrix(output_dir + 'iso_post/backscatter_iso.dat',self.backscatter_iso_post)
        write_ascii_isotropic_matrix(output_dir + 'iso_post/bnoise_iso.dat',self.bnoise_iso_post)
        write_ascii_isotropic_matrix(output_dir + 'iso_post/evector_iso.dat',self.evector_iso_post)
        write_ascii_isotropic_vector(output_dir + 'iso_post/evalue_iso.dat',self.evalue_iso_post)
        
        write_ascii_anisotropic_matrix(output_dir + 'drain_meanfield_aniso.none.dat',self.drain*0.0)
        write_ascii_isotropic_matrix(output_dir + 'drain_meanfield_iso.none.dat',self.drain_iso*0.0)
        
        write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.none.L01.dat',self.f_subgrid_avg[0,:,:]*0.0)
        write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.none.L02.dat',self.f_subgrid_avg[1,:,:]*0.0)

        write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.all.L01.dat',self.f_subgrid_avg[0,:,:]*scale_by)
        write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.all.L02.dat',self.f_subgrid_avg[1,:,:]*scale_by)

        
        if self.read_meanfields:
            write_ascii_anisotropic_matrix(output_dir + 'drain_meanfield_aniso.dat',self.D_meanfield)
            write_ascii_isotropic_matrix(output_dir + 'drain_meanfield_iso.dat',self.D_meanfield_iso)

            self.vort_q_aniso_avg = convert_q_to_vort(self.FL, self.q_avg)
            write_ascii_field_2d(output_dir + 'vort.q.L01.dat',self.vort_q_aniso_avg[0,:,:]*self.omega)
            write_ascii_field_2d(output_dir + 'vort.q.L02.dat',self.vort_q_aniso_avg[1,:,:]*self.omega)      

            self.vort_f_aniso_avg = convert_q_to_vort(self.FL, self.f_avg)
            write_ascii_field_2d(output_dir + 'vort.f.L01.dat',self.vort_f_aniso_avg[0,:,:]*scale_by)
            write_ascii_field_2d(output_dir + 'vort.f.L02.dat',self.vort_f_aniso_avg[1,:,:]*scale_by)

            self.vort_b_aniso_avg = convert_q_to_vort(self.FL, self.b_avg)
            write_ascii_field_2d(output_dir + 'vort.b.L01.dat',self.vort_b_aniso_avg[0,:,:]*scale_by)
            write_ascii_field_2d(output_dir + 'vort.b.L02.dat',self.vort_b_aniso_avg[1,:,:]*scale_by)

            self.vort_D_q = convert_q_to_vort(self.FL, self.D_q_avg)
            write_ascii_field_2d(output_dir + 'vort.neg_Dq.L01.dat',-self.vort_D_q[0,:,:]*scale_by)
            write_ascii_field_2d(output_dir + 'vort.neg_Dq.L02.dat',-self.vort_D_q[1,:,:]*scale_by)

            self.vort_chi_h = convert_q_to_vort(self.FL, self.chi_h_avg)
            write_ascii_field_2d(output_dir + 'vort.chiH.L01.dat',self.vort_chi_h[0,:,:]*scale_by)
            write_ascii_field_2d(output_dir + 'vort.chiH.L02.dat',self.vort_chi_h[1,:,:]*scale_by)

            self.vort_chi = convert_q_to_vort(self.FL, self.chi_avg)
            write_ascii_field_2d(output_dir + 'vort.chi.L01.dat',self.vort_chi[0,:,:]*self.omega)
            write_ascii_field_2d(output_dir + 'vort.chi.L02.dat',self.vort_chi[1,:,:]*self.omega)

            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.b.L01.dat',self.b_avg[0,:,:]*scale_by)
            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.b.L02.dat',self.b_avg[1,:,:]*scale_by)

            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.negDq.L01.dat',-self.D_q_avg[0,:,:]*scale_by)
            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.negDq.L02.dat',-self.D_q_avg[1,:,:]*scale_by)

            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.chiH.L01.dat',self.chi_h_avg[0,:,:]*scale_by)
            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.chiH.L02.dat',self.chi_h_avg[1,:,:]*scale_by)

            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.negDq_chiH.L01.dat',(self.chi_h_avg[0,:,:]-self.D_q_avg[0,:,:])*scale_by)
            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.negDq_chiH.L02.dat',(self.chi_h_avg[1,:,:]-self.D_q_avg[1,:,:])*scale_by)

            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.b_chiH.L01.dat',(self.b_avg[0,:,:]+self.chi_h_avg[0,:,:])*scale_by)
            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.b_chiH.L02.dat',(self.b_avg[1,:,:]+self.chi_h_avg[1,:,:])*scale_by)

            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.b_negDq.L01.dat',(self.b_avg[0,:,:]-self.D_q_avg[0,:,:])*scale_by)
            write_ascii_field_2d(output_dir + 'red_vort_subgrid_tend.avg_trunc.b_negDq.L02.dat',(self.b_avg[1,:,:]-self.D_q_avg[1,:,:])*scale_by)

        else:
            print('Meanfields not yet read from file.')
        

        return

    #---------------------------------------------------------------------------    
    def convert_vector_vort_to_q(self, i,F,data_baro):
        M = np.array(np.zeros((2,2), dtype=np.complex))
        if (i==0):
            c = 1.0/(2.0+4.0*F/(float(i)+1.0)/(float(i)+2.0))
        else:
            c = 1.0/(2.0+4.0*F/float(i)/(float(i)+1.0))
        M[0,0] = 0.5 + c
        M[0,1] = 0.5 - c
        M[1,0] = 0.5 - c
        M[1,1] = 0.5 + c
        data = np.array(np.zeros((2), dtype=np.complex))
        data = np.dot(M,data_baro)
        return data
    
    #---------------------------------------------------------------------------    
    def convert_vector_baro_to_q(self, i,F,data_baro):
        M = np.array(np.zeros((2,2), dtype=np.complex))
        if (i==0):
            c = 1.0/(2.0+4.0*F/(float(i)+1.0)/(float(i)+2.0))
        else:
            c = 1.0/(2.0+4.0*F/float(i)/(float(i)+1.0))
        M[0,0] = 1.0
        M[0,1] = 0.5/c
        M[1,0] = 1.0
        M[1,1] = -0.5/c
        data = np.array(np.zeros((2), dtype=np.complex))
        data = np.dot(M,data_baro)
        return data

    #---------------------------------------------------------------------------    
    def convert_vector_q_to_baro(self, i,F,data):
        M = np.array(np.zeros((2,2), dtype=np.complex))
        data_baro = np.array(np.zeros((2), dtype=np.complex))
        if (i==0):
            c = 1.0/(2.0+4.0*F/(float(i)+1.0)/(float(i)+2.0))
        else:
            c = 1.0/(2.0+4.0*F/float(i)/(float(i)+1.0))
        M[0,0] = 0.5
        M[0,1] = 0.5
        M[1,0] = c
        M[1,1] = -c
        data_baro = np.dot(M,data)
        return data_baro

    #---------------------------------------------------------------------------    
    def convert_evis_q_to_baro(self, i,F,data):
        M = np.array(np.zeros((2,2), dtype=np.complex))
        data_baro = np.array(np.zeros((2,2), dtype=np.complex))
        if (i==0):
            c = 1.0/(2.0+4.0*F/(float(i)+1.0)/(float(i)+2.0))
        else:
            c = 1.0/(2.0+4.0*F/float(i)/(float(i)+1.0))
        M[0,0] = 0.5
        M[0,1] = 0.5
        M[1,0] = c
        M[1,1] = -c
        data_baro = np.dot(M,np.dot(data,linalg.inv(M)))
        return data_baro

    #---------------------------------------------------------------------------    
    def convert_evis_vort_to_q(self, i,F,data_vort):
        M = np.array(np.zeros((2,2), dtype=np.complex))
        data_baro = np.array(np.zeros((2,2), dtype=np.complex))
        if (i==0):
            c = 1.0/(2.0+4.0*F/(float(i)+1.0)/(float(i)+2.0))
        else:
            c = 1.0/(2.0+4.0*F/float(i)/(float(i)+1.0))
        M[0,0] = 0.5 + c
        M[0,1] = 0.5 - c
        M[1,0] = 0.5 - c
        M[1,1] = 0.5 + c
        data = np.dot(linalg.inv(M),np.dot(data_vort,M))
        return data

    #---------------------------------------------------------------------------    
    def convert_evis_baro_to_q(self, i,F,data_baro):
        M = np.array(np.zeros((2,2), dtype=np.complex))
        data = np.array(np.zeros((2,2), dtype=np.complex))
        if (i==0):
            c = 1.0/(2.0+4.0*F/(float(i)+1.0)/(float(i)+2.0))
        else:
            c = 1.0/(2.0+4.0*F/float(i)/(float(i)+1.0))
        M[0,0] = 0.5
        M[0,1] = 0.5
        M[1,0] = c
        M[1,1] = -c
        data = np.dot(linalg.inv(M),np.dot(data_baro,M))
        return data

    #---------------------------------------------------------------------------    
    def convert_stats_q_to_baro(self, i,F,data):
        M = np.array(np.zeros((2,2), dtype=np.complex))
        if (i==0):
            c = 1.0/(2.0+4.0*F/(float(i)+1.0)/(float(i)+2.0))
        else:
            c = 1.0/(2.0+4.0*F/float(i)/(float(i)+1.0))
        M[0,0] = 0.5
        M[0,1] = 0.5
        M[1,0] = c
        M[1,1] = -c
        data_baro = np.array(np.zeros((2,2), dtype=np.complex))
        data_baro = np.dot(M,np.dot(data,M.conj().T))
        return data_baro

    #---------------------------------------------------------------------------    
    def convert_stats_baro_to_q(self, i,F,data_baro):
        M = np.array(np.zeros((2,2), dtype=np.complex))
        if (i==0):
            c = 1.0/(2.0+4.0*F/(float(i)+1.0)/(float(i)+2.0))
        else:
            c = 1.0/(2.0+4.0*F/float(i)/(float(i)+1.0))
        M[0,0] = 1.0
        M[0,1] = 0.5/c
        M[1,0] = 1.0
        M[1,1] = -0.5/c
        data = np.array(np.zeros((2,2), dtype=np.complex))
        data = np.dot(M,np.dot(data_baro,M.conj().T))
        return data

    #---------------------------------------------------------------------------    
    def convert_vector_vort_to_baro(self, data_vort):
        M = np.array(np.ones((2,2), dtype=np.complex))
        M[1,1] = -1.0
        M = M*0.5
        data_baro = np.array(np.zeros((2), dtype=np.complex))
        data_baro = np.dot(M,data_vort)
        return data_baro

    #---------------------------------------------------------------------------    
    def convert_vector_baro_to_vort(self, data_baro):
        M = np.array(np.ones((2,2), dtype=np.complex))
        M[1,1] = -1.0
        M = M*0.5
        data_vort = np.array(np.zeros((2), dtype=np.complex))
        data_vort = np.dot(2.0*M,data_baro)
        return data_vort

    #---------------------------------------------------------------------------    
    def convert_stats_baro_to_vort(self, data_baro):
        M = np.array(np.ones((2,2), dtype=np.complex))
        M[1,1] = -1.0
        M = M*0.5
        data_vort = np.array(np.zeros((2,2), dtype=np.complex))
        data_vort = np.dot(2.0*M,np.dot(data_baro,2.0*M))
        return data_vort

    #---------------------------------------------------------------------------    
    def convert_evis_baro_to_vort(self, data_baro):
        M = np.array(np.ones((2,2), dtype=np.complex))
        M[1,1] = -1.0
        M = M*0.5
        data_vort = np.array(np.zeros((2,2), dtype=np.complex))
        data_vort = np.dot(2.0*M,np.dot(data_baro,M))
        return data_vort