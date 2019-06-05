from math import log, sqrt, floor
from numpy import exp
import pdb
import numpy as np
import pdb

def wrapper(t, y):
    tDrugApplication = 10000
    INaFRedMed = 1
    ICaLRedMed = 1
    IKrRedMed = 1
    IKsRedMed = 1
    
    return Paci2018(t,y, tDrugApplication, INaFRedMed,
             ICaLRedMed, IKrRedMed, IKsRedMed)

def Paci2018(time, Y, tDrugApplication, INaFRedMed,
             ICaLRedMed, IKrRedMed, IKsRedMed):

    dY = np.empty(23)

    '''
    Parameters from optimizer   
      VmaxUp    = param(1)
      g_irel_max  = param(2)
      RyRa1         = param(3)
      RyRa2         = param(4)
      RyRahalf      = param(5)
      RyRohalf      = param(6)
      RyRchalf      = param(7)
      kNaCa         = param(8)
      PNaK          = param(9)
      Kup     = param(10)
      V_leak    = param(11)
      alpha         = param(12)
    '''
    VmaxUp = 0.5113      # millimolar_per_second (in calcium_dynamics)
    g_irel_max = 62.5434 # millimolar_per_second (in calcium_dynamics)
    RyRa1 = 0.05354      # uM
    RyRa2 = 0.0488       # uM
    RyRahalf = 0.02427   # uM
    RyRohalf = 0.01042   # uM
    RyRchalf = 0.00144   # uM
    kNaCa = 3917.0463    # A_per_F (in i_NaCa)
    PNaK = 2.6351        # A_per_F (in i_NaK)
    Kup = 3.1928e-4      # millimolar (in calcium_dynamics)
    V_leak = 4.7279e-4   # per_second (in calcium_dynamics)
    alpha = 2.5371       # dimensionless (in i_NaCa)

    ## Constants
    F = 96485.3415   # coulomb_per_mole (in model_parameters)
    R = 8.314472   # joule_per_mole_kelvin (in model_parameters)
    T = 310.0   # kelvin (in model_parameters)

    ## Cell geometry
    V_SR = 583.73   # micrometre_cube (in model_parameters)
    Vc   = 8800.0   # micrometre_cube (in model_parameters)
    Cm   = 9.87109e-11   # farad (in model_parameters)

    ## Extracellular concentrations
    Nao = 151.0   # millimolar (in model_parameters)
    Ko  = 5.4   # millimolar (in model_parameters)
    Cao = 1.8 #3#5#1.8   # millimolar (in model_parameters)

    ## Intracellular concentrations
    # Naio = 10 mM Y[17]
    Ki = 150.0   # millimolar (in model_parameters)
    # Cai  = 0.0002 mM Y[2]
    # caSR = 0.3 mM Y[1]

    # time (second)

    ## Nernst potential
    E_Na = R*T/F*log(Nao/Y[17])

    E_Ca = 0.5*R*T/F*log(Cao/Y[2])

    E_K  = R*T/F*log(Ko/Ki)
    PkNa = 0.03   # dimensionless (in electric_potentials)
    E_Ks = R*T/F*log((Ko+PkNa*Nao)/(Ki+PkNa*Y[17]))


    ## INa
    g_Na        = 3671.2302   # S_per_F (in i_Na)
    i_Na        = ((time<tDrugApplication)*1+(time >= tDrugApplication)*INaFRedMed)*g_Na*Y[13]**3.0*Y[11]*Y[12]*(Y[0]-E_Na)

    h_inf       = 1.0/sqrt(1.0+exp((Y[0]*1000.0+72.1)/5.7))
    alpha_h     = 0.057*exp(-(Y[0]*1000.0+80.0)/6.8)
    beta_h      = 2.7*exp(0.079*Y[0]*1000.0)+3.1*10.0**5.0*exp(0.3485*Y[0]*1000.0)
    if (Y[0] < -0.0385):
        tau_h   = 1.5/((alpha_h+beta_h)*1000.0)
    else:
        tau_h   = 1.5*1.6947/1000.0
    
    dY[11]   = (h_inf-Y[11])/tau_h

    j_inf       = 1.0/sqrt(1.0+exp((Y[0]*1000.0+72.1)/5.7))
    if (Y[0] < -0.04):
        alpha_j = (-25428.0*exp(0.2444*Y[0]*1000.0)-6.948*10.0**-6.0*exp(-0.04391*Y[0]*1000.0))*(Y[0]*1000.0+37.78)/(1.0+exp(0.311*(Y[0]*1000.0+79.23)))
    else:
        alpha_j = 0.0
    
    if (Y[0] < -0.04):
        beta_j  = ((0.02424*exp(-0.01052*Y[0]*1000)/(1+exp(-0.1378*(Y[0]*1000+40.14)))))
    else:
        beta_j  = ((0.6*exp((0.057)*Y[0]*1000)/(1+exp(-0.1*(Y[0]*1000+32)))))
    
    tau_j       = 7.0/((alpha_j+beta_j)*1000.0)
    dY[12]   = (j_inf-Y[12])/tau_j

    m_inf       = 1.0/(1.0+exp((-Y[0]*1000.0-34.1)/5.9))**(1.0/3.0)
    alpha_m     = 1.0/(1.0+exp((-Y[0]*1000.0-60.0)/5.0))
    beta_m      = 0.1/(1.0+exp((Y[0]*1000.0+35.0)/5.0))+0.1/(1.0+exp((Y[0]*1000.0-50.0)/200.0))
    tau_m       = 1.0*alpha_m*beta_m/1000.0
    dY[13]   = (m_inf-Y[13])/tau_m



    ## INaL
    myCoefTauM  = 1
    tauINaL     = 200 #ms
    GNaLmax     = 2.3*7.5 #(S/F)
    Vh_hLate    = 87.61
    i_NaL       = GNaLmax* Y[18]**(3)*Y[19]*(Y[0]-E_Na)

    m_inf_L     = 1/(1+exp(-(Y[0]*1000+42.85)/(5.264)))
    alpha_m_L   = 1/(1+exp((-60-Y[0]*1000)/5))
    beta_m_L    = 0.1/(1+exp((Y[0]*1000+35)/5))+0.1/(1+exp((Y[0]*1000-50)/200))
    tau_m_L     = 1/1000 * myCoefTauM*alpha_m_L*beta_m_L
    dY[18]   = (m_inf_L-Y[18])/tau_m_L

    h_inf_L     = 1/(1+exp((Y[0]*1000+Vh_hLate)/(7.488)))
    tau_h_L     = 1/1000 * tauINaL
    dY[19]   = (h_inf_L-Y[19])/tau_h_L

    ## If
    E_f         = -0.017   # volt (in i_f)
    g_f         = 30.10312   # S_per_F (in i_f)

    i_f         = g_f*Y[14]*(Y[0]-E_f)
    i_fNa       = 0.42*g_f*Y[14]*(Y[0]-E_Na)

    Xf_infinity = 1.0/(1.0+exp((Y[0]*1000.0+77.85)/5.0))
    tau_Xf      = 1900.0/(1.0+exp((Y[0]*1000.0+15.0)/10.0))/1000.0
    dY[14]   = (Xf_infinity-Y[14])/tau_Xf




    ## ICaL
    g_CaL       = 8.635702e-5   # metre_cube_per_F_per_s (in i_CaL)
    i_CaL       = ((time<tDrugApplication)*1+(time >= tDrugApplication)*ICaLRedMed)*g_CaL*4.0*Y[0]*F**2.0/(R*T)*(Y[2]*exp(2.0*Y[0]*F/(R*T))-0.341*Cao)/(exp(2.0*Y[0]*F/(R*T))-1.0)*Y[4]*Y[5]*Y[6]*Y[7]

    d_infinity  = 1.0/(1.0+exp(-(Y[0]*1000.0+9.1)/7.0))
    alpha_d     = 0.25+1.4/(1.0+exp((-Y[0]*1000.0-35.0)/13.0))
    beta_d      = 1.4/(1.0+exp((Y[0]*1000.0+5.0)/5.0))
    gamma_d     = 1.0/(1.0+exp((-Y[0]*1000.0+50.0)/20.0))
    tau_d       = (alpha_d*beta_d+gamma_d)*1.0/1000.0
    dY[4]    = (d_infinity-Y[4])/tau_d

    f1_inf      = 1.0/(1.0+exp((Y[0]*1000.0+26.0)/3.0))
    if (f1_inf-Y[5] > 0.0):
        constf1 = 1.0+1433.0*(Y[2]-50.0*1.0e-6)
    else:
        constf1 = 1.0
    
    tau_f1      = (20.0+1102.5*exp(-((Y[0]*1000.0+27.0)**2.0/15.0)**2.0)+200.0/(1.0+exp((13.0-Y[0]*1000.0)/10.0))+180.0/(1.0+exp((30.0+Y[0]*1000.0)/10.0)))*constf1/1000.0
    dY[5]    = (f1_inf-Y[5])/tau_f1

    f2_inf      = 0.33+0.67/(1.0+exp((Y[0]*1000.0+32.0)/4.0))
    constf2     = 1.0
    tau_f2      = (600.0*exp(-(Y[0]*1000.0+25.0)**2.0/170.0)+31.0/(1.0+exp((25.0-Y[0]*1000.0)/10.0))+16.0/(1.0+exp((30.0+Y[0]*1000.0)/10.0)))*constf2/1000.0
    dY[6]    = (f2_inf-Y[6])/tau_f2

    alpha_fCa   = 1.0/(1.0+(Y[2]/0.0006)**8.0)
    beta_fCa    = 0.1/(1.0+exp((Y[2]-0.0009)/0.0001))
    gamma_fCa   = 0.3/(1.0+exp((Y[2]-0.00075)/0.0008))
    fCa_inf     = (alpha_fCa+beta_fCa+gamma_fCa)/1.3156
    if ((Y[0] > -0.06) and (fCa_inf > Y[7])):
        constfCa = 0.0
    else:
        constfCa = 1.0
    
    tau_fCa     = 0.002   # second (in i_CaL_fCa_gate)
    dY[7]    = constfCa*(fCa_inf-Y[7])/tau_fCa

    ## Ito
    g_to        = 29.9038   # S_per_F (in i_to)
    i_to        = g_to*(Y[0]-E_K)*Y[15]*Y[16]

    q_inf       = 1.0/(1.0+exp((Y[0]*1000.0+53.0)/13.0))
    tau_q       = (6.06+39.102/(0.57*exp(-0.08*(Y[0]*1000.0+44.0))+0.065*exp(0.1*(Y[0]*1000.0+45.93))))/1000.0
    dY[15]   = (q_inf-Y[15])/tau_q

    r_inf       = 1.0/(1.0+exp(-(Y[0]*1000.0-22.3)/18.75))
    tau_r       = (2.75352+14.40516/(1.037*exp(0.09*(Y[0]*1000.0+30.61))+0.369*exp(-0.12*(Y[0]*1000.0+23.84))))/1000.0
    dY[16]   = (r_inf-Y[16])/tau_r

    ## IKs
    g_Ks        = 2.041   # S_per_F (in i_Ks)
    i_Ks        = ((time<tDrugApplication)*1+(time >= tDrugApplication)*IKsRedMed)*g_Ks*(Y[0]-E_Ks)*Y[10]**2.0*(1.0+0.6/(1.0+(3.8*0.00001/Y[2])**1.4))

    Xs_infinity = 1.0/(1.0+exp((-Y[0]*1000.0-20.0)/16.0))
    alpha_Xs    = 1100.0/sqrt(1.0+exp((-10.0-Y[0]*1000.0)/6.0))
    beta_Xs     = 1.0/(1.0+exp((-60.0+Y[0]*1000.0)/20.0))
    tau_Xs      = 1.0*alpha_Xs*beta_Xs/1000.0
    dY[10]   = (Xs_infinity-Y[10])/tau_Xs

    ## IKr
    L0           = 0.025   # dimensionless (in i_Kr_Xr1_gate)
    Q            = 2.3   # dimensionless (in i_Kr_Xr1_gate)
    g_Kr         = 29.8667   # S_per_F (in i_Kr)
    i_Kr         = ((time<tDrugApplication)*1+(time >= tDrugApplication)*IKrRedMed)*g_Kr*(Y[0]-E_K)*Y[8]*Y[9]*sqrt(Ko/5.4)

    V_half       = 1000.0*(-R*T/(F*Q)*log((1.0+Cao/2.6)**4.0/(L0*(1.0+Cao/0.58)**4.0))-0.019)

    Xr1_inf      = 1.0/(1.0+exp((V_half-Y[0]*1000.0)/4.9))
    alpha_Xr1    = 450.0/(1.0+exp((-45.0-Y[0]*1000.0)/10.0))
    beta_Xr1     = 6.0/(1.0+exp((30.0+Y[0]*1000.0)/11.5))
    tau_Xr1      = 1.0*alpha_Xr1*beta_Xr1/1000.0
    dY[8]     = (Xr1_inf-Y[8])/tau_Xr1

    Xr2_infinity = 1.0/(1.0+exp((Y[0]*1000.0+88.0)/50.0))
    alpha_Xr2    = 3.0/(1.0+exp((-60.0-Y[0]*1000.0)/20.0))
    beta_Xr2     = 1.12/(1.0+exp((-60.0+Y[0]*1000.0)/20.0))
    tau_Xr2      = 1.0*alpha_Xr2*beta_Xr2/1000.0
    dY[9]    = (Xr2_infinity-Y[9])/tau_Xr2

    ## IK1
    alpha_K1 = 3.91/(1.0+exp(0.5942*(Y[0]*1000.0-E_K*1000.0-200.0)))
    beta_K1  = (-1.509*exp(0.0002*(Y[0]*1000.0-E_K*1000.0+100.0))+exp(0.5886*(Y[0]*1000.0-E_K*1000.0-10.0)))/(1.0+exp(0.4547*(Y[0]*1000.0-E_K*1000.0)))
    XK1_inf  = alpha_K1/(alpha_K1+beta_K1)
    g_K1     = 28.1492   # S_per_F (in i_K1)
    i_K1     = g_K1*XK1_inf*(Y[0]-E_K)*sqrt(Ko/5.4)

    ## INaCa
    KmCa   = 1.38   # millimolar (in i_NaCa)
    KmNai  = 87.5   # millimolar (in i_NaCa)
    Ksat   = 0.1   # dimensionless (in i_NaCa)
    gamma  = 0.35   # dimensionless (in i_NaCa)
    kNaCa1 = kNaCa   # A_per_F (in i_NaCa)
    i_NaCa = kNaCa1*(exp(gamma*Y[0]*F/(R*T))*Y[17]**3.0*Cao-exp((gamma-1.0)*Y[0]*F/(R*T))*Nao**3.0*Y[2]*alpha)/((KmNai**3.0+Nao**3.0)*(KmCa+Cao)*(1.0+Ksat*exp((gamma-1.0)*Y[0]*F/(R*T))))

    ## INaK
    Km_K  = 1.0   # millimolar (in i_NaK)
    Km_Na = 40.0   # millimolar (in i_NaK)
    PNaK1 = PNaK   # A_per_F (in i_NaK)
    i_NaK = PNaK1*Ko/(Ko+Km_K)*Y[17]/(Y[17]+Km_Na)/(1.0+0.1245*exp(-0.1*Y[0]*F/(R*T))+0.0353*exp(-Y[0]*F/(R*T)))

    ## IpCa
    KPCa  = 0.0005   # millimolar (in i_PCa)
    g_PCa = 0.4125   # A_per_F (in i_PCa)
    i_PCa = g_PCa*Y[2]/(Y[2]+KPCa)







    ## Background currents
    g_b_Na = 0.95   # S_per_F (in i_b_Na)
    i_b_Na = g_b_Na*(Y[0]-E_Na)

    g_b_Ca = 0.727272   # S_per_F (in i_b_Ca)
    i_b_Ca = g_b_Ca*(Y[0]-E_Ca)

    ## Sarcoplasmic reticulum
    i_up = VmaxUp/(1.0+Kup**2.0/Y[2]**2.0)

    i_leak = (Y[1]-Y[2])*V_leak

    dY[3] = 0

    # RyR
    RyRSRCass = (1 - 1/(1 +  exp((Y[1]-0.3)/0.1)))
    i_rel = g_irel_max*RyRSRCass*Y[21]*Y[22]*(Y[1]-Y[2])

    RyRainfss = RyRa1-RyRa2/(1 + exp((1000*Y[2]-(RyRahalf))/0.0082))
    RyRtauadapt = 1 #s
    dY[20] = (RyRainfss- Y[20])/RyRtauadapt

    RyRoinfss = (1 - 1/(1 +  exp((1000*Y[2]-(Y[20]+ RyRohalf))/0.003)))
    if (RyRoinfss>= Y[21]):
      RyRtauact = 18.75e-3       #s
    else:
      RyRtauact = 0.1*18.75e-3   #s
    
    dY[21] = (RyRoinfss- Y[21])/RyRtauact

    RyRcinfss = (1/(1 + exp((1000*Y[2]-(Y[20]+RyRchalf))/0.001)))
    if (RyRcinfss>= Y[22]):
      RyRtauinact = 2*87.5e-3    #s
    else:
        RyRtauinact = 87.5e-3      #s
    
    dY[22] = (RyRcinfss- Y[22])/RyRtauinact

    ## Ca2+ buffering
    Buf_C       = 0.25   # millimolar (in calcium_dynamics)
    Buf_SR      = 10.0   # millimolar (in calcium_dynamics)
    Kbuf_C      = 0.001   # millimolar (in calcium_dynamics)
    Kbuf_SR     = 0.3   # millimolar (in calcium_dynamics)
    Cai_bufc    = 1.0/(1.0+Buf_C*Kbuf_C/(Y[2]+Kbuf_C)**2.0)
    Ca_SR_bufSR = 1.0/(1.0+Buf_SR*Kbuf_SR/(Y[1]+Kbuf_SR)**2.0)

    ## Ionic concentrations
    #Nai
    dY[17] = -Cm*(i_Na+i_NaL+i_b_Na+3.0*i_NaK+3.0*i_NaCa+i_fNa)/(F*Vc*1.0e-18)
    #caSR
    dY[2]  = Cai_bufc*(i_leak-i_up+i_rel-(i_CaL+i_b_Ca+i_PCa-2.0*i_NaCa)*Cm/(2.0*Vc*F*1.0e-18))
    #Cai
    dY[1]  = Ca_SR_bufSR*Vc/V_SR*(i_up-(i_rel+i_leak))

    ## Stimulation
    i_stim_Amplitude    = 7.5e-10   # ampere (in stim_mode)
    i_stim_End        = 1000.0   # second (in stim_mode)
    i_stim_PulseDuration  = 0.005   # second (in stim_mode)
    i_stim_Start      = 0.0   # second (in stim_mode)
    i_stim_frequency    = 60.0   # per_second (in stim_mode)
    stim_flag         = 0.0   # dimensionless (in stim_mode)
    i_stim_Period       = 60.0/i_stim_frequency

    if ((time >= i_stim_Start) and (time <= i_stim_End) and (time-i_stim_Start-floor((time-i_stim_Start)/i_stim_Period)*i_stim_Period <= i_stim_PulseDuration)):
        i_stim = stim_flag*i_stim_Amplitude/Cm
    else:
        i_stim = 0.0
    

    ## Membrane potential
    dY[0] = -(i_K1+i_to+i_Kr+i_Ks+i_CaL+i_NaK+i_Na+i_NaL+i_NaCa+i_PCa+i_f+i_b_Na+i_b_Ca-i_stim)

    ## Output variables
    IK1     = i_K1
    Ito     = i_to
    IKr     = i_Kr
    IKs     = i_Ks
    ICaL     = i_CaL
    INaK    = i_NaK
    INa     = i_Na
    INaCa   = i_NaCa
    IpCa    = i_PCa
    If      = i_f
    IbNa    = i_b_Na
    IbCa    = i_b_Ca
    Irel    = i_rel
    Iup     = i_up
    Ileak   = i_leak
    Istim   = i_stim
    INaL    = i_NaL

    dati = [INa, If, ICaL, Ito, IKs, IKr, IK1, INaCa, INaK, IpCa, IbNa, IbCa, Irel, Iup, Ileak, Istim, E_K, E_Na, INaL]

    return dY


# 1 - evaluate function at Y...
# 2 - loop over Y, perturb each element by a small percent of its total
# 3 - during each loop, find dY/dt, given the perturbation
# 4 - find the change in Y as a function of the change in the perturbation
