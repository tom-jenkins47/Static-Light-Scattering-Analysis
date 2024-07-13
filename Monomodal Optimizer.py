"""
Tom Jenkins s1946411
V2 03/03/23
Optimizes the form factor for a given intensity distribution, under the assumption that
the form factor is monomodal.
The optimziation method used self-determines the initial guesses.
A more accurate polydispersity initial guess if known may improve the quality of the optimization.

Input file:
.txt file of format angle[deg], intensity[a.u] (perp), intensity[a.u] (parallel), intensity[a.u] (non-polarized)
or
.txt file of format angle[deg] intensity[a.u] (perp)

In the second case, change lines 261 and 262 to;
angles = data[0::2]
intensities = data[1::2]
"""

import numpy as np
import matplotlib.pyplot as plt
import math 
import scipy.optimize
import scipy.linalg
from scipy.stats import norm
from scipy.signal import savgol_filter
from scipy.integrate import quad
import time



def wave_vector(n_solvent, wavelength, angle):
    """
    Parameters
    ----------
    n_solvent : refractive index of solvent
    wavelength : wavelength of probe beam in nm
    angle : measured scattering angle in radians

    Returns
    -------
    The calculated wavevector q.
    """
    return ((4*np.pi*n_solvent)/wavelength)*np.sin((angle*np.pi/180)/2)

def form_factor(q, R):
    """
    Parameters
    ----------
    q : calculated wavevector
    R : paticle size in microns

    Returns
    -------
    The calculated form factor.
    """
    return ((3/(q*R)**3)*(np.sin(q*R)-q*R*np.cos(q*R)))**2

def critical_points(array):
    """
    Parameters
    ----------
    array : array of intensity or form factor values

    Returns
    -------
    An array containing the indices of the minima and maxima of the input array.
    """
    
    inverse = -1*array
    min_ind, _ = scipy.signal.find_peaks(inverse)
    max_ind, _ = scipy.signal.find_peaks(array)
    
    return min_ind, max_ind

def inflexion_points(array):
    """
    Parameters
    ----------
    array : array of intensity or form factor values

    Returns
    -------
    An array containing the indices of the inflexion points of the array.
    """
    
    diff2 = np.gradient(np.gradient(array))
    return np.where(np.diff(np.sign(diff2)))[0]
    

def anomaly_suppression(mins, spacing):
    """
    Parameters
    ----------
    mins : array of minima indices
    spacing : minimum index separation, below which shall be attributed to an anomaly

    Returns
    -------
    new_mins : adjusted array of indices to account for any anomalous values.

    """
    
    for i in np.arange(1, len(mins)):
         if mins[i]-mins[i-1] < spacing:
             mins[i] = math.floor((mins[i]+mins[i-1])/2)
             new_mins = np.delete(mins, i-1)      
    return new_mins

def smoothing(array, n):
    """
    Parameters
    ----------
    array : array to be smoothed
    n : magnitude of smoothing parameter, higher magnitude results in a 
        smoother curve

    Returns
    -------
    Smoothed array.

    """
    return savgol_filter(array, n, 2)

def poly_integrand(R, q, R_av, sigma):
    """
    Parameters
    ----------
    R : Particle size
    q : Wavevector
    R_av : Guess of particle size
    sigma : Standard deviation of normal factor

    Returns
    -------
    The integrand of the polydisperse form factor.
    """
    return (((3/(q*R)**3)*(np.sin(q*R)-q*R*np.cos(q*R)))**2)*((R/(R_av-5*sigma))**6)*norm.pdf(R, R_av, sigma)


def R_estimator(smooth_check, q_vals, intensities, residual_check, inflexion_check):
    """
    Parameters
    ----------
    n_solvent : Refractive index of solvent
    wavelength_nm : Wavelength of probe beam in nm
    smooth_param : Magnitude of smoothing.
    angles : Array of experimental angle data
    intensities : Array of experimental intensity data

    Returns
    -------
    best_R : Estimate of best R value.
    q_vals : Array of q values generated from angle data
    """

    # Create intensity log plot subject to smoothing parameters
    if smooth_check == 'n':
        log_int = np.log(intensities)
    else:
        log_int = smoothing(np.log(intensities), 7)
    
    int_mins, int_maxes = critical_points(log_int)
    int_inflexions = inflexion_points(log_int)
    
    R = (0.001)*1e-6
    
    R_vals = []
    
    # Cycle through values of R to check if number of critical points match
    while R<2e-6:
    
        form_vals = np.zeros(len(q_vals))
        for i in range(len(q_vals)):
            form_vals[i] = form_factor(q_vals[i], R)
            
        log_form = np.log(form_vals)
        
        if inflexion_check == 'y':
            form_inflexions = inflexion_points(log_form)
            if len(int_inflexions) == len(form_inflexions):
                R_vals.append(R)
                R+=(0.001)*1e-6
            else:
                R+=(0.001)*1e-6
    
        if inflexion_check == 'n' and residual_check == 'y':
            form_mins, form_maxes = critical_points(log_form)
            if len(int_mins) == len(form_mins):
                R_vals.append(R)
                R+=(0.001)*1e-6
            else:
                R+=(0.001)*1e-6
            
        if inflexion_check == 'n' and residual_check ==  'n': 
            form_mins, form_maxes = critical_points(log_form)
            if len(int_mins) == len(form_mins) and len(int_maxes) == len(form_maxes):
                R_vals.append(R)
                R+=(0.001)*1e-6
            else:
                R+=(0.001)*1e-6
    
    R_vals = np.around(np.array(R_vals), 9)
    
    residuals = []
    
    # Roughly find value of R that minimises the residuals
    for R in R_vals:
        
        form_vals = np.zeros(len(q_vals))
        for i in range(len(q_vals)):
            form_vals[i] = form_factor(q_vals[i], R)
            
        log_form = np.log(form_vals)         
        
        if inflexion_check == 'y':
            form_inflexions = inflexion_points(log_form)
            residual = np.linalg.norm(int_inflexions - form_inflexions)
            residuals.append(residual)
            
        if inflexion_check == 'n' and residual_check == 'y':
            form_mins, form_maxes = critical_points(log_form)
            residual = np.linalg.norm(int_mins-form_mins)
            residuals.append(residual)
            
        if inflexion_check == 'n' and residual_check == 'n':
            form_mins, form_maxes = critical_points(log_form)
            residual = np.linalg.norm(int_mins-form_mins) + np.linalg.norm(int_maxes-form_maxes)
            residuals.append(residual)
        
    best_R_ind = np.argmin(residuals)
    best_R = np.round(R_vals[best_R_ind], 9)
        
    for i in range(len(q_vals)):
        form_vals[i] = form_factor(q_vals[i], best_R)
        
    log_form = np.log(form_vals)
    
    intensities_norm = intensities/intensities[0]
    form_norm = form_vals/form_vals[0]
    
    plt.plot(q_vals*best_R, np.log(form_norm), label='form factor')
    plt.plot(q_vals*best_R, np.log(intensities_norm), label='intensity distribution')
    plt.xlabel("qR")
    plt.ylabel("P(q, R)")
    plt.title("E1 Pre-Optimization Normalized Intensity and Form Factor Distribution (Log)")
    plt.legend()
    plt.show()
    
    return best_R
    
def main():
    
    # Metadata
    n_solvent = 1.36
    wavelength_nm = 632.8
    
    # Load data file
    data = np.loadtxt("ASSi91.txt").flatten()
    
    # Extract intensity and angle data
    angles = data[0::2]
    intensities = data[1::2]
    
    # Normalize the intensities
    int_norm = intensities/intensities[0]

    wavelength = wavelength_nm*1e-9
    
    # Create array of wavevectors
    q_vals = np.zeros(len(angles))
    for i in range(len(angles)):
        q_vals[i] = wave_vector(n_solvent, wavelength, angles[i])
        
    
    # Plot input intensity array to check form 
    plt.plot(q_vals, np.log(int_norm))
    plt.title('Initial Plot')
    plt.xlabel('Wave Vector')
    plt.ylabel('Intensity')
    plt.show()
    
    # Form checks to improve data analysis and optimization
    residual_check = str(input("Is there only one minima? [y/n]: "))
    smooth_check = str(input("Does the plot need smoothing? [y/n]: "))
    inflexion_check = str(input("Does the plot contain inflexion points? [y/n]: "))
    
    # If minima depth is not strictly decreasing or there are inflexion points, suspect the sample is bimodal
    init_mins, _  = critical_points(np.log(int_norm))
    diff_list = np.diff(np.log(int_norm)[init_mins])
    if np.all(diff_list < 0) == False or inflexion_check == 'y':
        print('\nWARNING: The sample is likely to be bimodal')
    
    # Guess initial size value using manual optimization
    R_guess = R_estimator(smooth_check, q_vals, intensities, residual_check, inflexion_check)
    # The guess shouldn't matter too much given the optimization on this parameter is unbounded
    poly_guess = 8
    sigma_guess = (poly_guess/100)*R_guess
    
    
    # Optimization function that determines the residuals of the form factor and intensity distribution
    # curve_fit does not work due to coupled x-axis dependence on q and R
    def poly_form_factor_calc(params):
        sigma, R_av = params
        poly_form_factor = np.zeros(len(q_vals))
        residual = np.zeros(len(q_vals))
        
        # Set up integration limits
        R_min = R_av - 5*sigma
        R_max = R_av + 5*sigma
     
        # Below is a standard chisq minimization
        # The error of 1e-9 aligns with the experimental error
        for i in range(len(q_vals)):
            poly_form_factor[i] = quad(poly_integrand, R_min, R_max, args=(q_vals[i], R_av, sigma))[0]
            residual[i] = (np.log(int_norm[i]) - np.log(poly_form_factor[i]/poly_form_factor[0]))/np.log(1e-9)
        return (np.linalg.norm(residual))**2
    
    # Time the optimization process
    t1 = time.time()
    
    # Bound the optimization values  
    bnds = ((0, np.inf), (0.9*R_guess, 1.1*R_guess))
    
    # Nelder-Mead uses a gradient-free approach and can also take bounds with latest version of Scipy
    # Maxiter argument can be added to constrain optimization duration
    res = scipy.optimize.minimize(poly_form_factor_calc, [sigma_guess, R_guess], method='Nelder-Mead', bounds=bnds, options={'disp' : True})

    t2 = time.time()
    
    # Return the final optimization values
    sigma_final = (res.x)[0]
    R_final = (res.x)[1]
    
    form_factor_final = np.zeros(len(q_vals))
    
    # Finalize the integration limits
    R_min_final = R_final - 5*sigma_final
    R_max_final = R_final + 5*sigma_final
    
    # Plot the final polydisperse, monomodal form factor
    for i in range(len(q_vals)):
        form_factor_final[i] = quad(poly_integrand, R_min_final, R_max_final, args=(q_vals[i], R_final, sigma_final))[0]
    
    # Determine the final polydispersity
    polydispersity = 100*(sigma_final/R_final)
    
    print(f'\nThe particle size is {R_final/(1e-6)} microns')
    print(f'The polydispersity is {polydispersity}%')
    print(f'\nThe optimization took {(t2-t1)} seconds.')
    
    # Below was used to determine the limits of RGD
    
    """
    err_list1 = []
    m_vals1 = []
    n_set = np.array([1.01, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0])
  
    for n in n_set:
        R_form = np.zeros(len(q_vals))
        q_form = np.zeros(len(q_vals))
        Mie_data = np.loadtxt(f"{n}test.txt").flatten()
        Mie_angles = Mie_data[0::4]
        Mie_ints = Mie_data[1::4]
        for i in range(len(q_vals)):
            q_form[i] = wave_vector(n, 632e-9, Mie_angles[i])
            R_form[i] = form_factor(q_form[i], 1e-6)
        R_form_norm = R_form/np.max(R_form)
        Mie_ints_norm = Mie_ints/np.max(Mie_ints)
        Mie_residuals = np.zeros(len(q_vals))
        plt.plot(q_form, np.log(R_form_norm))
        plt.plot(q_form, np.log(Mie_ints_norm))
        plt.show()
        for i in range(len(q_vals)):
            Mie_residuals[i] = (np.log(R_form_norm[i]) - np.log(Mie_ints_norm[i]))
        err_list1.append(((np.linalg.norm(Mie_residuals))**2)/119)
        m_vals1.append(1-(1.0/n))
        
    err_list2 = []
    m_vals2 = []
    R_set = np.arange(200, 1250, 50)
    
    for R in R_set:
        R_form = np.zeros(len(q_vals))
        q_form = np.zeros(len(q_vals))
        Mie_data = np.loadtxt(f"{R}test.txt").flatten()
        Mie_angles = Mie_data[0::4]
        Mie_ints = Mie_data[1::4]
        for i in range(len(q_vals)):
            q_form[i] = wave_vector(1.1, 632e-9, Mie_angles[i])
            R_form[i] = form_factor(q_form[i], R*1e-9)
        R_form_norm = R_form/np.max(R_form)
        Mie_ints_norm = Mie_ints/np.max(Mie_ints)
        Mie_residuals = np.zeros(len(q_vals))
        plt.plot(q_form, np.log(R_form_norm))
        plt.plot(q_form, np.log(Mie_ints_norm))
        plt.show()
        for i in range(len(q_vals)):
            Mie_residuals[i] = (np.log(R_form_norm[i]) - np.log(Mie_ints_norm[i]))
        err_list2.append(((np.linalg.norm(Mie_residuals))**2)/119)
        tol = ((2*np.pi*1.1*R*1e-9)/(632e-9))*(1-(1/1.1))
        m_vals2.append(tol)
        
    plt.plot(m_vals1, err_list1)
    plt.title("Influence of Moving Further Outwith the RGD Regime on $\chi^{2}$ (1)")
    plt.xlabel(f"|1-m|")
    plt.ylabel(f"Normalized $\chi^{2}$")
    plt.show()
        
    plt.plot(m_vals2, err_list2)
    plt.title("Influence of Moving Further Outwith the RGD Regime on $\chi^{2}$ (2)")
    plt.xlabel(f"$((4 \pi n_s a) / \lambda)$|1-m|")
    plt.ylabel(f"Normalized $\chi^{2}$")
    plt.show()
    """
    
    """
    form_fact_chi_sq = np.zeros(len(q_vals))
    residual_chi_sq = np.zeros(len(q_vals))
    for i in range(len(q_vals)):
        form_fact_chi_sq[i] = quad(poly_integrand, 0, 1e-6, args=(q_vals[i], 0.29860385742187523e-6, 2.081023633e-8))[0]
        residual_chi_sq[i] = np.log(int_norm[i]) - np.log(form_fact_chi_sq[i]/form_fact_chi_sq[0])
    chi_sq = (np.linalg.norm(residual_chi_sq))**2   
    print(f"The value of chi sq is {chi_sq}")
    """

    plt.plot(q_vals*R_final, np.log(form_factor_final/form_factor_final[0]), label='form factor')
    plt.plot(q_vals*R_final, np.log(int_norm), label='intensity distribution')
    plt.title("S3 Final Normalized/Optimized Intensity and Form Factor Distribution (Log)")
    plt.xlabel("qR")
    plt.ylabel("P(q, $\sigma$, $R_{av}$)")
    plt.legend()
    plt.show()
    
    plt.plot(q_vals*R_final, form_factor_final/form_factor_final[0], label='form factor')
    plt.plot(q_vals*R_final, int_norm, label='intensity distribution')
    plt.title("S3 Final Normalized/Optimized Intensity and Form Factor Distribution")
    plt.xlabel("qR")
    plt.ylabel("P(q, $\sigma$, $R_{av}$)")
    plt.legend()
    plt.show() 
        
if __name__ == '__main__':
    main()
    
    
    

    
    
    
    
    


    
    
    

