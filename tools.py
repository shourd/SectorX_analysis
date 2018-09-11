"""
AUTHOR: TJITTE DE JONG
EDITED BY: SJOERD VAN ROOIJEN
"""

# This file contains functions that are used in other scripts in this project

import math

# String to boolean function
def strToBool(string):
    if string.lower() == "true":
        return True
    elif string.lower() == "false":
        return False
    else:
        raise ValueError

# Convert feet to nautical miles
def ftToNm(distance):
    return distance * 0.000164578834

# Convert nautical miles to feet
def nmToFt(distance):
    return distance / 0.000164578834

###############################################################################
# function to calculate Closest Point of Approach (CPA) properties in 2D
# INPUT: 
#    position, speed magnitude, and direction of two vehicles
#
# OUTPUT:
#    miss distance [m], time-to-CPA [s], and the location
#    of both vehicles where the CPA occurs [m].
###############################################################################
def getCPA( pos_a, v_a, dir_a_deg, pos_b, v_b, dir_b_deg ) :
    
    # declare and init output variables
    CPA_s = 0.0
    CPA_m = 0.0
    where_a = [0.0,0.0]
    where_b = [0.0,0.0]
    
    # setup the speed vectors
    vel_a = [v_a*math.sin(math.radians(dir_a_deg)), v_a*math.cos(math.radians(dir_a_deg))]
    vel_b = [v_b*math.sin(math.radians(dir_b_deg)), v_b*math.cos(math.radians(dir_b_deg))]
    
    # calculate delta distance [m]
    dx = pos_b[0] - pos_a[0]
    dy = pos_b[1] - pos_a[1]
    
    # calculate delta speed [m/s]
    dvx = vel_b[0] - vel_a[0]
    dvy = vel_b[1] - vel_a[1]
    
    # calculate the 2D CPA stuff
    if( not(dvx == 0.0 and dvy == 0.0) ) :
    
        CPA_s = - (dx * dvx + dy * dvy ) / ( dvx * dvx + dvy * dvy )    
        
        CPA_m = math.sqrt(
    		(dx + dvx*CPA_s) *
    		(dx + dvx*CPA_s) +
    		(dy + dvy*CPA_s) *
    		(dy + dvy*CPA_s)
    		)
      
        where_a[0] = pos_a[0] + vel_a[0] * CPA_s
        where_a[1] = pos_a[1] + vel_a[1] * CPA_s
        
        where_b[0] = pos_b[0] + vel_b[0] * CPA_s
        where_b[1] = pos_b[1] + vel_b[1] * CPA_s
      
    else :
       
        CPA_s = 0.0
        CPA_m = math.sqrt( dx * dx + dy * dy)
        where_a = [0.0,0.0]
        where_b = [0.0,0.0]
   
    
    return (CPA_m, CPA_s, where_a, where_b)
###############################################################################

###############################################################################
# function to convert IAS to VTAS using International Standard Atmosphere (ISA)
###############################################################################
def ISA_IAStoTAS_kts(IAS_kts, alt_ft) :
    
    KTS_2_MS   = 0.51444444
    rho0_kg_m3 = 1.225
    P0_Pa      = 101325
    gamma      = 1.4
    mu         = (gamma-1)/gamma    
    
    # first, get the pressure at the alt
    P       = _getP(alt_ft)  
    rho	= _getRho(alt_ft)  
    IAS_ms  = IAS_kts*0.51444444
    A       =  1 + (mu/2)*(rho0_kg_m3/P0_Pa)*IAS_ms*IAS_ms
    B       =  math.pow(A,1/mu) - 1
    C       =  B * P0_Pa/P  + 1
    D       =  math.pow(C,mu)   - 1
    TAS_ms  =  math.sqrt(D*(2/mu)*(P/rho))
    
    return(TAS_ms/KTS_2_MS)
###############################################################################

###############################################################################
# ISA helper function
###############################################################################
def _getRho(alt_ft) :
    
    TROPALT_FT = 36100	# troposphere start flightlevel (ft)
    FT_2_M     = 0.3048
    g_m_s2     = 9.81
    R          = 287.0
    T0_K       = 288.15
    rho0_kg_m3 = 1.225
    rho_exp    = 4.26
    
    if (alt_ft > TROPALT_FT) :
        
        return (_getRho(TROPALT_FT) * math.exp(-(g_m_s2/( R * _getT(TROPALT_FT)))* FT_2_M * (alt_ft-TROPALT_FT)))
    else :
        
        return (rho0_kg_m3 * math.pow((_getT(alt_ft)/T0_K),rho_exp))
###############################################################################
        
###############################################################################
# ISA helper function
###############################################################################
def _getT(alt_ft) :
    
    FT_2_M     = 0.3048
    TROPALT_FT = 36100	# troposphere start flightlevel (ft)
    T0_K       = 288.15
    d_T_K_m    = -0.0065
    
    if (alt_ft > (TROPALT_FT)) :
        
        return (T0_K + d_T_K_m * FT_2_M * TROPALT_FT)
        
    else :
        
        return (T0_K + d_T_K_m * FT_2_M * alt_ft)
###############################################################################
        
###############################################################################
# ISA helper function
###############################################################################
def _getP(alt_ft) :
    
    FT_2_M     = 0.3048
    TROPALT_FT = 36100	# troposphere start flightlevel (ft)
    g_m_s2     = 9.81
    R          = 287.0
    P0_Pa      = 101325
    P_exp      = 5.26
    T0_K       = 288.15
    
    if (alt_ft > TROPALT_FT) :
        
        return _getP(TROPALT_FT) * math.exp(-(g_m_s2/(R*_getT(TROPALT_FT))) * FT_2_M * (alt_ft-TROPALT_FT))

    else :
        
        return (P0_Pa*math.pow((_getT(alt_ft)/T0_K),P_exp))
###############################################################################
