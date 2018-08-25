# -*- coding: utf-8 -*-
"""
Wave propagation module
------------------------

Wave propagation in elastic media
"""
from __future__ import division, print_function
from numpy import cos, sin, exp, pi, arcsin, sqrt
from numpy import array


#%% Classical elasticity
def scatter_matrix(alpha, beta, ang_i, ang_j):
    r"""
    Scatter matrix for reflection/conversion coefficients
    in an elastic halfspace
    
    The matrix is written as presented in [AKI2009]_.
    
    .. math::

        \begin{bmatrix}
            \acute{P}\grave{P} &\acute{P}\grave{S}\\
            \acute{S}\grave{P} &\acute{S}\grave{S}
        \end{bmatrix}

    Parameters
    ----------
    alpha : float
        Speed for the P-wave.
    beta : float
        Speed for the S-wave.
    ang_i : ndarray
        Incidence angle for the P-wave.
    ang_j : ndarray
        Incidence angle for the S-wave.

    Returns
    -------
    scatter : ndarray
        Scatter matrix for the reflection conversion modes.

    References
    ----------
    
    .. [AKI2009] Keiiti Aki and Paul G. Richards. Quantitative
        Seismology. University Science Books, 2009.
    """
    p = sin(ang_i)/alpha
    p1 = cos(ang_i)/alpha
    p2 = cos(ang_j)/beta
    denom = (1/beta**2 - 2*p**2)**2 + 4*p**2*p1*p2
    PP = -(1/beta**2 - 2*p**2)**2 + 4*p**2*p1*p2
    PS = 4*alpha/beta*p*p1*(1/beta**2 - 2*p**2)
    SP = 4*beta/alpha*p*p2*(1/beta**2 - 2*p**2)
    SS = -PP
    scatter = array([
                [PP/denom, PS/denom],
                [SP/denom, SS/denom]])
    return scatter



def disp_incident_P(amp, omega, alpha, beta, ang_i, x, z):
    """Displacement for a P-wave incidence

     Parameters
    ----------
    amp : ndarray
        Amplitude for given frequencies.
    omega : ndarray
        Angular frequency.
    alpha : float
        Speed of the P wave.
    beta : float
        Speed of the S wave.
    ang_i : float
        Incidence angle.
    x : ndarray
        Horizontal coordinate.
    z : ndarray
        Vertical coordinate.

    Returns
    -------
    u : ndarray
        Spectrum for the horizontal component of displacement.
    v : ndarray
        Spectrum for the vertical component of displacement.
    
    """
    p = sin(ang_i)/alpha
    ang_j = arcsin(beta * p)
    p1 = cos(ang_i)/alpha
    p2 = cos(ang_j)/beta
    scatter = scatter_matrix(alpha, beta, ang_i, ang_j) 
    PP = scatter[0, 0]
    PS = scatter[0, 1]
    
    # Horizontal
    u_P_in = amp * sin(ang_i) * exp(1j*omega*(p*x - p1*z))
    u_P_ref = amp * sin(ang_i) * PP * exp(1j*omega*(p*x + p1*z))
    u_S_ref = amp * cos(ang_j) * PS * exp(1j*omega*(p*x + p2*z))
    
    # Vertical
    v_P_in = -amp * cos(ang_i) * exp(1j*omega*(p*x - p1*z))
    v_P_ref = amp * cos(ang_i) * PP * exp(1j*omega*(p*x + p1*z))
    v_S_ref = -amp * sin(ang_j) * PS * exp(1j*omega*(p*x + p2*z))
    
    # Total
    u = u_P_in + u_P_ref + u_S_ref
    v = v_P_in + v_P_ref + v_S_ref
    return u, v


#%% Micropolar elasticity
def scatter_matrix_micropolar(c1, c2, v2, K, ang_i, ang_j):
    r"""
    Scatter matrix for reflection/conversion coefficients
    in an micropolar halfspace
    

    Parameters
    ----------
    c1 : float
        Speed for the P-wave.
    c2 : float
        Speed for the classical S-wave.
    v2 : float
        Speed for the S-wave.
    K : float
        New speed in micropolar.
    ang_i : ndarray
        Incidence angle for the P-wave.
    ang_j : ndarray
        Incidence angle for the S-wave.

    Returns
    -------
    scatter : ndarray
        Scatter matrix for the reflection conversion modes.

    """
    p = sin(ang_i)/c1
    p1 = cos(ang_i)/c1
    p2 = cos(ang_j)/v2
    denom = ((1/c2**2 - 2*p**2)*(1/v2**2 - 2*p**2) - K**2/(2*c2**2*v2**2)*
             (1/c2**2 - 2*p**2)) + 4*p**2*p1*p2
    PP = -((1/c2**2 - 2*p**2)*(1/v2**2 - 2*p**2) - K**2/(2*c2**2*v2**2)*
             (1/c2**2 - 2*p**2)) + 4*p**2*p1*p2           
    PS = 4*c1/v2*p*p1*(1/c2**2 - 2*p**2)
    SS = -PP
    SP = 4*v2/c1*p*p2*(1/c2**2 - 2*p**2) - 2*p*p2*K**2/(v2**2*c2**2)
    scatter = array([
                [PP/denom, PS/denom],
                [SP/denom, SS/denom]])
    return scatter


def wavenumber_micropolar(c2, c4, Q, K, omega):
    r"""
    Dispersion relations for for shear and rotational waves in micropolar
    media as presented in [NOW]_.
    

    Parameters
    ----------
    c2 : float
        Phase speed for the S-wave in the high frequency limit.
    c4 : float
        Phase speed for the R-wave in the high frequency limit.
    Q : float
        Micropolar parameter. The cut-off frequency for the rotational
        wave is given by :math:`\omega_0^2 = 2 Q^2`.
    K : float
        Micropolar parameter.
    omega : ndarray
        Angular frequency.


    Returns
    -------
    kappa_S : ndarray
        Wavenumber for the shear wave.
    kappa_R : ndarray
        Wavenumber for the rotational wave.
    
    References
    ----------
    .. [NOW] Witold Nowacki. Theory of micropolar elasticity.
        International centre for mechanical sciences,
        Courses and lectures, No. 25. Berlin: Springer, 1972.
    """
    D = omega**2/c4**2 + omega**2/c2**2 + K**2*Q**2/(c2**2*c4**2) - 2*Q**2/c4**2
    E = omega**4/(c2**2*c4**2) - 2*Q**2*omega**2/(c2**2*c4**2)
    kappa_S = sqrt(0.5*D + 0.5*sqrt(D**2 - 4*E))
    kappa_R = sqrt(0.5*D - 0.5*sqrt(D**2 - 4*E))
    return kappa_S, kappa_R


def frequency_micropolar(c2, c4, Q, K, kappa):
    r"""
    Dispersion relations for for shear and rotational waves in micropolar
    media as presented in [NOW]_.
    

    Parameters
    ----------
    c2 : float
        Phase speed for the S-wave in the high frequency limit.
    c4 : float
        Phase speed for the R-wave in the high frequency limit.
    Q : float
        Micropolar parameter. The cut-off frequency for the rotational
        wave is given by :math:`\omega_0^2 = 2 Q^2`.
    K : float
        Micropolar parameter.
    kappa : ndarray
        Wavenumber.


    Returns
    -------
    omega_S : ndarray
        Angular frequency for the shear wave.
    omega_R : ndarray
        Angular frequency for the rotational wave.
    
    References
    ----------
    .. [NOW] Witold Nowacki. Theory of micropolar elasticity.
        International centre for mechanical sciences,
        Courses and lectures, No. 25. Berlin: Springer, 1972.
    """
    A = 2*Q**2+ (c2**2 + c4**2)*kappa**2
    B = 2*Q**2*c2**2*kappa**2 - K**2*Q**2*kappa**2 + c2**2*c4**2*kappa**4
    omega_S = sqrt(0.5*A - 0.5*sqrt(A**2 - 4*B))
    omega_R = sqrt(0.5*A + 0.5*sqrt(A**2 - 4*B))
    return omega_S, omega_R


def phase_speed_micropolar(c2, c4, Q, K, omega):
    r"""
    Phase speed for shear and rotational waves in micropolar
    media as presented in [NOW]_.
    
    The phase speed is given by

    .. math::
        
        v_{S/R} = \frac{\omega}{\kappa_{S/R}}\, ,
    
    with
    
    .. math::
        
        \kappa_{S/R} = \sqrt{\frac{D}{2} \pm \frac{1}{2}\sqrt{D^2 - 4E}}\, ,
    
    and
    
    .. math::
    
        &D = \frac{1}{c_2^2 c_4^2}\left[(c_2^2 + c_4^2)\omega^2 - 2Q^2\left(c_2^2 - \frac{K^2}{2}\right)\right]\, ,\\
        &E = \frac{\omega^4}{c_2^2 c_4^2} - \frac{2Q^2 \omega^2}{c_2^2 c_4^2}\, .


    Parameters
    ----------
    c2 : float
        Phase speed for the S-wave in the high frequency limit.
    c4 : float
        Phase speed for the R-wave in the high frequency limit.
    Q : float
        Micropolar parameter. The cut-off frequency for the rotational
        wave is given by :math:`\omega_0^2 = 2 Q^2`.
    K : float
        Micropolar parameter.
    omega : ndarray
        Angular frequency.


    Returns
    -------
    v_S : ndarray
        Phase speed for the shear wave.
    v_R : ndarray
        Phase speed for the rotational wave.
    
    References
    ----------
    .. [NOW] Witold Nowacki. Theory of micropolar elasticity.
        International centre for mechanical sciences,
        Courses and lectures, No. 25. Berlin: Springer, 1972.
    """
    kappa_S, kappa_R = wavenumber_micropolar(c2, c4, Q, K, omega)
    v_S = omega/kappa_S
    v_R = omega/kappa_R
    return v_S, v_R


def disp_incident_P_micropolar(amp, omega, c1, c2, c4, Q, K, ang_i, x, z):
    """Displacement for a P-wave incidence

     Parameters
    ----------
    amp : ndarray
        Amplitude for given frequencies.
    omega : ndarray
        Angular frequency.
    alpha : float
        Speed of the P wave.
    beta : float
        Speed of the S wave.
    ang_i : float
        Incidence angle.
    x : ndarray
        Horizontal coordinate.
    z : ndarray
        Vertical coordinate.

    Returns
    -------
    u : ndarray
        Spectrum for the horizontal component of displacement.
    v : ndarray
        Spectrum for the vertical component of displacement.
    
    """
    p = sin(ang_i + 0j)/c1
    p1 = cos(ang_i + 0j)/c1
    v2, _ = phase_speed_micropolar(c2, c4, Q, K, omega)
    ang_j = arcsin(v2 * p)
    p2 = cos(ang_j)/v2
    scatter = scatter_matrix_micropolar(c1, c2, v2, K, ang_i, ang_j)
    PP = scatter[0, 0]
    PS = scatter[0, 1]
    
    # Horizontal
    u_P_in = amp * sin(ang_i) * exp(1j*omega*(p*x - p1*z))
    u_P_ref = amp * sin(ang_i) * PP * exp(1j*omega*(p*x + p1*z))
    u_S_ref = amp * cos(ang_j) * PS * exp(1j*omega*(p*x + p2*z))
    
    # Vertical
    v_P_in = -amp * cos(ang_i) * exp(1j*omega*(p*x - p1*z))
    v_P_ref = amp * cos(ang_i) * PP * exp(1j*omega*(p*x + p1*z))
    v_S_ref = -amp * sin(ang_j) * PS * exp(1j*omega*(p*x + p2*z))
    
    # Total
    u = u_P_in + u_P_ref + u_S_ref
    v = v_P_in + v_P_ref + v_S_ref
    return u, v


#%% CST elasticty
def scatter_matrix_cst(c1, c2, v2, L, omega, ang_i, ang_j):
    r"""
    Scatter matrix for reflection/conversion coefficients
    in an micropolar halfspace
    

    Parameters
    ----------
    c1 : float
        Speed for the P-wave.
    c2 : float
        Speed for the classical S-wave.
    v2 : float
        Speed for the S-wave.
    L : float
        Length scale in CST.
    omega : ndarray
        Angular frequency.
    ang_i : ndarray
        Incidence angle for the P-wave.
    ang_j : ndarray
        Incidence angle for the S-wave.

    Returns
    -------
    scatter : ndarray
        Scatter matrix for the reflection conversion modes.

    """
    p = sin(ang_i)/c1
    p1 = cos(ang_i)/c1
    p2 = cos(ang_j)/v2
    denom = (1/c2**2 - 2*p**2)*(1/v2**2 - 2*p**2 + 2*L**2*omega**2/v2**4) + 4*p**2*p1*p2
    PP = -(1/c2**2 - 2*p**2)*(1/v2**2 - 2*p**2 + 2*L**2*omega**2/v2**4) + 4*p**2*p1*p2
    PS = -4*c1/v2*p*p1*(1/c2**2 - 2*p**2)
    SS = -PP
    SP = 4*v2/c1*p*p2*(1/v2**2 - 2*p**2 + 2*L**2*omega**2/v2**4)
    scatter = array([
                [PP/denom, PS/denom],
                [SP/denom, SS/denom]])
    return scatter


def frequency_cst(c2, k, L):
    r"""
    Angular frequency for shear wave in Couple-Stress-Theory as
    presented in [CST]_.
    
    The amngular frequency is given by

    .. math::
        
        \omega_S = c_2 \kappa \sqrt{1 + \kappa^2 l^2 }


    Parameters
    ----------
    c2 : float
        Speed for the classical S-wave.
    k : ndarray
        Wavenumber.
    L : float
        Length scale in CST.

    Returns
    -------
    omega : ndarray
        Angular frequency for the shear wave.
    
    References
    ----------
    .. [CST] Ali R. Hadhesfandiari, Gary F. Dargush.
        Couple stress theory for solids. International Journal
        for Solids and Structures, 2011, 48, 2496-2510.
    """
    return c2*k*sqrt(1 + k**2*L**2)
    

def phase_speed_cst(c2, omega, L):
    r"""
    Phase speed for shear wave in Couple-Stress-Theory as
    presented in [CST]_.
    
    The phase speed is given by

    .. math::
        
        v_S = c_2\sqrt{1 + \frac{1}{2}\left[\sqrt{1 + \frac{4l^2\omega^2}{c_2^2}}  - 1\right]}


    Parameters
    ----------
    c2 : float
        Speed for the classical S-wave.
    omega : ndarray
        Angular frequency.
    L : float
        Length scale in CST.

    Returns
    -------
    v2 : ndarray
        Phase speed for the shear wave.
    
    References
    ----------
    .. [CST] Ali R. Hadhesfandiari, Gary F. Dargush.
        Couple stress theory for solids. International Journal
        for Solids and Structures, 2011, 48, 2496-2510.
    """
    return c2*sqrt(1 + 0.5*(sqrt(1 + 4*L**2*omega**2/c2**2) - 1))



def disp_incident_P_cst(amp, omega, c1, c2, Lc, ang_i, x, z):
    """Displacement for a P-wave incidence

     Parameters
    ----------
    amp : ndarray
        Amplitude for given frequencies.
    omega : ndarray
        Angular frequency.
    alpha : float
        Speed of the P wave.
    beta : float
        Speed of the S wave.
    ang_i : float
        Incidence angle.
    x : ndarray
        Horizontal coordinate.
    z : ndarray
        Vertical coordinate.

    Returns
    -------
    u : ndarray
        Spectrum for the horizontal component of displacement.
    v : ndarray
        Spectrum for the vertical component of displacement.
    
    """
    p = sin(ang_i + 0j)/c1
    p1 = cos(ang_i + 0j)/c1
    v2 = phase_speed_cst(c2, omega, Lc)
    ang_j = arcsin(v2 * p)
    p2 = cos(ang_j)/v2
    scatter = scatter_matrix_cst(c1, c2, v2, Lc, omega, ang_i, ang_j)
    PP = scatter[0, 0]
    PS = scatter[0, 1]
    
    # Horizontal
    u_P_in = amp * sin(ang_i) * exp(1j*omega*(p*x - p1*z))
    u_P_ref = amp * sin(ang_i) * PP * exp(1j*omega*(p*x + p1*z))
    u_S_ref = amp * cos(ang_j) * PS * exp(1j*omega*(p*x + p2*z))
    
    # Vertical
    v_P_in = -amp * cos(ang_i) * exp(1j*omega*(p*x - p1*z))
    v_P_ref = amp * cos(ang_i) * PP * exp(1j*omega*(p*x + p1*z))
    v_S_ref = -amp * sin(ang_j) * PS * exp(1j*omega*(p*x + p2*z))
    
    # Total
    u = u_P_in + u_P_ref + u_S_ref
    v = v_P_in + v_P_ref + v_S_ref
    return u, v    


#%% Signals
def ricker_spectrum(freq, freq_c):
    r"""Spectrum of the Ricker wavelet
    
    The spectrum is given by
    
    .. math::
        
        \hat{f}(\xi) = -\xi^2 e^{-\xi^2}

    where :math:`\xi= f/f_c` is the dimensionless frequency, and 
    :math:`f_c` is the central frequency. The time between peaks
    is given by
    
    .. math::

        t_\text{peaks} = \frac{\sqrt{6}}{\pi f_c}\, .
 
    For further reference see [KIM1991]_ and [RIC1945]_.
    
    Parameters
    ----------
    freq : ndarray
        Frequency.
    freq_c : float
        Frequency for the peak.

    Returns
    -------
    amplitude : ndarray
        Fourier amplitude for the given frequencies.

    References
    ----------
    
    .. [KIM1991] A. Papageorgiou and J. Kim. Study of the propagation
        of seismic waves in Caracas Valley with reference to the
        29 July 1967 earthquake: SH waves. Bulletin of the
        Seismological Society of America, 1991, 81 (6): 2214-223.
    .. [RIC1945] N. Ricker. The computation of output disturbances from
        from amplifiers for true wavelets inputs. Geophysics,
        10: 207-220.
    """
    xi = freq/freq_c
    return -xi**2*exp(-xi**2)


def ricker_signal(t, freq_c):
    r"""Ricker wavelet in the time domain
    
    The signal is given by
    
    .. math::
        
        f(\tau) = (2\tau^2 - 1) e^{-\tau^2}

    where :math:`\tau=\pi f_c t` is the dimensionless time, and
    :math:`f_c` is the central frequency. The time between peaks
    is given by
    
    .. math::

        t_\text{peaks} = \frac{\sqrt{6}}{\pi f_c}\, .
 
    For further reference see [KIM1991]_.
    
    Parameters
    ----------
    t : ndarray
        Time.
    freq_c : float
        Central frequency for the signal.

    Returns
    -------
    signal : ndarray
        Signal evaluated at given times.

    References
    ----------
    
    .. [KIM1991] A. Papageorgiou and J. Kim. Study of the propagation
        of seismic waves in Caracas Valley with reference to the
        29 July 1967 earthquake: SH waves. Bulletin of the
        Seismological Society of America, 1991, 81 (6): 2214-223.
    .. [RIC1945] N. Ricker. The computation of output disturbances from
        from amplifiers for true wavelets inputs. Geophysics,
        10: 207-220.
    """
    tau = pi*t*freq_c
    return (2*tau**2 - 1)*exp(-tau**2)



if __name__ ==  "__main__":
    import doctest
    doctest.testmod()
