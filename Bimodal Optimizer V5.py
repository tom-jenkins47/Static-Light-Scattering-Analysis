"""
Tom Jenkins s1946411
V5 01/03/23
Optimizes the form factor for a given intensity distribution, under the assumption that
the form factor is bimodal.
The optimziation method used requires initial guesses for the poly dispersities and particle radii.
More accurate initial guesses improve the quality of the optimization.
Polydispersity guesses should be within 50% of the correct values for simulated data.

Input file:
.txt file of format angle[deg], intensity[a.u] (perp), intensity[a.u] (parallel), intensity[a.u] (non-polarized)
or
.txt file of format angle[deg] intensity[a.u] (perp)

In the second case, change lines 188 and 189 to;
angles = data[0::2]
intensities = data[1::2]
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.linalg
from scipy.stats import norm
from scipy.signal import savgol_filter
from scipy.integrate import quad
import time


def wave_vector(n_solvent, wavelength, angle):
    """
    Computes the wavevector for a given angle
    """  
    return ((4*np.pi*n_solvent)/wavelength)*np.sin((angle*np.pi/180)/2)

def form_factor(q, R):
    """
    Computes the monodisperse, monomodal form factor for a given wavevector
    """
    return ((3/(q*R)**3)*(np.sin(q*R)-q*R*np.cos(q*R)))**2

def critical_points(array): 
    """
    Finds the index locations of the critical points of an array
    """
    inverse = -1*array
    min_ind, _ = scipy.signal.find_peaks(inverse)
    max_ind, _ = scipy.signal.find_peaks(array)
    return min_ind, max_ind

def inflexion_points(array):
    """
    Finds the index locations of the inflexion points of an array
    This method is less accurate than critical_points
    """
    diff2 = np.gradient(np.gradient(array))
    return np.where(np.diff(np.sign(diff2)))[0]

def smoothing(array, n):
    """
    Smooths a noisy array subject to an arbritary smoothing parameter
    """
    return savgol_filter(array, n, 2)

def poly_integrand(R, q, R_av, sigma):
    """
    Computes the polydisperse, monomodal form factor integrand
    """
    return (((3/(q*R)**3)*(np.sin(q*R)-q*R*np.cos(q*R)))**2)*((R/(R_av-5*sigma))**6)*norm.pdf(R, R_av, sigma)

def poly_integrand_bi(R, q, R_av1, R_av2, sigma1, sigma2, n1, n2):
    """
    Computes the polydisperse, bimodal form factor integrand
    """
    return (((3/(q*R)**3)*(np.sin(q*R)-q*R*np.cos(q*R)))**2)*((R/(R_av2-5*sigma2))**6)\
           *(1/(n1+n2))*(n1*norm.pdf(R, R_av1, sigma1)+n2*norm.pdf(R, R_av2, sigma2))

def R_estimator(smooth_check, q_vals, intensities, residual_check, inflexion_check):
    """
    Estimates the largest particle radii corresponding to an intensity distribution
    """
    
    # Check whether function requires smoothing
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
    plt.title("Pre-Optimization Normalized Intensity and Form Factor Distribution (Log)")
    plt.xlabel("qR")
    plt.ylabel("P(q, R)")
    plt.legend()
    plt.show()
    
    return best_R
    
def main():

    # Metadata
    n_solvent = 1.37
    wavelength_nm = 632.8
    
    # Load data file
    data = np.loadtxt("ASM365.txt").flatten()
    
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
    
    """
    sim_curve = np.zeros(len(q_vals))
    for i in range(len(q_vals)):
        sim_curve[i] =  quad(poly_integrand_bi, 0.195e-6 , 1.44e-6, args=(q_vals[i], 1.2e-6, 0.3e-6, 4.44e-8, 3.6e-8, 1, 1))[0]
    """
    # Plot input intensity array to check form
    plt.plot(q_vals, np.log(int_norm), label = 'intensity distribution')
    #plt.plot(q_vals, np.log(sim_curve/sim_curve[0]), label = 'theoretical RGD fit')
    plt.title('Initial Plot')
    plt.xlabel('q [$m^{-1}$]')
    plt.ylabel('P(q, $\sigma_1$, $\sigma_2$, $R_{av1}$, $R_{av2}$, $n_1$, $n_2$)')
    plt.legend()
    plt.show()
    
    # Form checks to improve data analysis and optimization
    residual_check = str(input("Is there only one minima? [y/n]: "))
    smooth_check = str(input("Does the plot need smoothing? [y/n]: "))
    inflexion_check = str(input("Does the plot contain inflexion points? [y/n]: "))
    num_ratio = str(input("Are there more small particles than large? [y/n]: "))
    n_guess = float(input("What is roughly the relevant number concentration?: ")) # if there are more small this is n1
    size_ratio = int(input("What is the approximate ratio of particle radii?: ")) # i.e what is R1 divided by R2
    
    # If minima depth is not strictly decreasing or there are inflexion points, suspect the sample is bimodal
    init_mins, _  = critical_points(np.log(int_norm))
    diff_list = np.diff(np.log(int_norm)[init_mins])
    if np.all(diff_list < 0) == False or inflexion_check == 'y':
        print('\nWARNING: The sample is likely to be bimodal')
     
    # Set up initial parameters, based on rough data from TEM measurements
    R_guess1 = R_estimator(smooth_check, q_vals, intensities, residual_check, inflexion_check)
    poly_guess1 = 5
    R_guess2 = R_guess1/size_ratio
    poly_guess2 = 10
    sigma_guess1 = (poly_guess1/100)*R_guess1
    sigma_guess2 = (poly_guess2/100)*R_guess2
    
    # Optimization function that determines the residuals of the form factor and intensity distribution
    # curve_fit does not work due to coupled x-axis dependence on q and R
    def poly_form_factor_calc(params):
        
        if num_ratio == 'y':
            R_av1, R_av2, sigma1, sigma2, n1 = params
            n2 = 1
        if num_ratio == 'n':
            R_av1, R_av2, sigma1, sigma2, n2 = params
            n1 = 1
        
        poly_form_factor = np.zeros(len(q_vals))
        residual = np.zeros(len(q_vals))
        
        # Maybe solidify these integration limits or move back outside function 
        R_min = R_av2 - 5*sigma2
        R_max = R_av1 + 5*sigma1
     
        # Below is a standard chisq minimization
        # The error of 1e-9 aligns with the experimental error
        for i in range(len(q_vals)):
            poly_form_factor[i] = quad(poly_integrand_bi, R_min, R_max, args=(q_vals[i], R_av1, R_av2, sigma1, sigma2, n1, n2))[0]
            residual[i] = (np.log(int_norm[i]) - np.log(poly_form_factor[i]/poly_form_factor[0]))/np.log(1e-9)
        return (np.linalg.norm(residual))**2
    
    # Time the optimization
    t1 = time.time()
        
    # Bound the optimization values
    # 25% size tolerances and 10% number concentrations are typical of a single rough TEM measurement
    bnds = ((0.75*R_guess1, 1.25*R_guess1), (0.75*R_guess2, 1.25*R_guess2), (0, np.inf), (0, np.inf), (0.9*n_guess, 1.1*n_guess))
    
    # Nelder-Mead uses a gradient-free approach and can also take bounds with latest version of Scipy
    # Maxiter argument can be added to constrain optimization duration
    res = scipy.optimize.minimize(poly_form_factor_calc, [R_guess1, R_guess2, sigma_guess1, sigma_guess2, n_guess], method='Nelder-Mead', bounds=bnds, options={'disp' : True})

    t2 = time.time()
    
    # Return the final optimization values
    R1_final = (res.x)[0]
    R2_final = (res.x)[1]
    sigma1_final = (res.x)[2]
    sigma2_final = (res.x)[3]
    n_final = (res.x)[4]
    
    form_factor_final = np.zeros(len(q_vals))
    
    # Finalize the integration limits
    R_min_final = R2_final - 5*sigma2_final
    R_max_final = R1_final + 5*sigma1_final

    
    # Plot the final polydisperse, bimodal form factor
    if num_ratio ==  'y':
        for i in range(len(q_vals)):
            form_factor_final[i] = quad(poly_integrand_bi, R_min_final, R_max_final, args=(q_vals[i], R1_final, R2_final, sigma1_final, sigma2_final, n_final, 1))[0]
        
    if num_ratio == 'n':
        for i in range(len(q_vals)):
            form_factor_final[i] = quad(poly_integrand_bi, R_min_final, R_max_final, args=(q_vals[i], R1_final, R2_final, sigma1_final, sigma2_final, 1, n_final))[0]

    # Determine final polydispersity values
    polydispersity1 = 100*(sigma1_final/R1_final)
    polydispersity2 = 100*(sigma2_final/R2_final)
    
    
    print(f'\nThe larger particle radius is {R1_final/(1e-6)} microns.')
    print(f'The larger polydispersity is {polydispersity1}%.')
    print(f'\nThe smaller particle radius is {R2_final/(1e-6)} microns.')
    print(f'The smaller polydispersity is {polydispersity2}%.')
    print(f'\nThe relative concentration of larger/smaller particles is {n_final}.')
    print(f'\nThe optimization took {(t2-t1)/60} minutes.')
    
    plt.plot(q_vals*R1_final, np.log(form_factor_final/form_factor_final[0]), label='bimodal form factor')
    plt.plot(q_vals*R1_final, np.log(int_norm), label='intensity distribution')
    plt.title("S6 Normalized/Optimized Bimodal Intensity and Form Factor Distribution (Log)")
    plt.xlabel("qR")
    plt.ylabel("P(q, $\sigma_1$, $\sigma_2$, $R_{av1}$, $R_{av2}$, $n_1$, $n_2$)")
    plt.legend()
    plt.show()
    

    plt.plot(q_vals*R1_final, form_factor_final/form_factor_final[0], label='bimodal form factor')
    plt.plot(q_vals*R1_final, int_norm, label='intensity distribution')
    plt.title("S6 Normalized/Optimized Bimodal Intensity and Form Factor Distribution")
    plt.xlabel("qR")
    plt.ylabel("P(q, $\sigma_1$, $\sigma_2$, $R_{av1}$, $R_{av2}$, $n_1$, $n_2$)")
    plt.legend()
    plt.show() 
  
if __name__ == '__main__':
    main()
    
    
    

    
    
    
    
    


    
    
    

