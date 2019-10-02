from math import log, sqrt
from typing import List
from cell_model import CellModel

import numpy as np
from scipy import integrate

import ga_configs
import protocols
import trace
from math import log, exp


class KernikModel(CellModel):
    """An implementation of the Kernik model by Kernik et al.

    Attributes:
        default_parameters: A dict containing tunable parameters along with
            their default values as specified in Kernik et al.
        updated_parameters: A dict containing all parameters that are being
            tuned.
    """

    # Load model parameters
    model_parameter_inputs = np.loadtxt('model_data/model_inputs_kernik.csv')

    # Current parameter values:
    x_K1 = model_parameter_inputs[16:22]
    x_KR = model_parameter_inputs[22:33]
    x_IKS = model_parameter_inputs[33:39]
    xTO = model_parameter_inputs[39:50]
    x_cal = model_parameter_inputs[50:61]
    x_cat = model_parameter_inputs[61]
    x_NA = model_parameter_inputs[62:76]
    x_F = model_parameter_inputs[76:82]

    Cm = 60
    V_tot = 3960
    Vc_tenT = 16404
    VSR_tenT = 1094
    V_tot_tenT = Vc_tenT + VSR_tenT
    Vc = V_tot * (Vc_tenT / V_tot_tenT)
    V_SR = V_tot * (VSR_tenT / V_tot_tenT)

    # Constants
    T = 310.0  # kelvin (in model_parameters)'
    R = 8.314472  # joule_per_mole_kelvin (in model_parameters)
    F = 96.4853415  # coulomb_per_mmole (in model_parameters)

    Ko = 5.4  # millimolar (in model_parameters)
    Cao = 1.8  # millimolar (in model_parameters
    Nao = 140.0  # millimolar (in model_parameters)

    def __init__(self, default_parameters=None,
                 updated_parameters=None,
                 no_ion_selective_dict=None,
                 default_time_unit='ms', 
                 default_voltage_unit='mV'
                 ):

        if not default_parameters:
            # default_parameters = self.model_parameter_inputs[0:16]
            default_parameters = {
                'G_K1': 1,
                'G_Kr': 1,
                'G_Ks': 1,
                'G_to': 1,
                'P_CaL': 1,
                'G_CaT': 1,
                'G_Na': 1,
                'G_F': 1,
                'K_NaCa': 1,
                'P_NaK': 1,
                'VmaxUp': 1,
                'V_leak': 1,
                'ks': 1,
                'G_b_Na': 1,
                'G_b_Ca': 1,
                'G_PCa': 1
            }

        y_initial = np.loadtxt('model_data/y_initial.csv')
        
        super().__init__(y_initial, default_parameters,
                         updated_parameters,
                         no_ion_selective_dict,
                         default_time_unit,
                         default_voltage_unit)

    def action_potential_diff_eq(self, t, y):
        """
        differential equations for Kernik iPSC-CM model
        solved by ODE15s in main_ipsc.m

        # State variable definitions:
        # 1: Vm (millivolt)

        # Ionic Flux: ---------------------------------------------------------
        # 2: Ca_SR (millimolar)
        # 3: Cai (millimolar)
        # 4: Nai (millimolar)
        # 5: Ki (millimolar)
        # 6: Ca_ligand (millimolar)

        # Current Gating (dimensionless):--------------------------------------
        # 7: d     (activation in i_CaL)
        # 8: f1    (inactivation in i_CaL)
        # 9: fCa   (calcium-dependent inactivation in i_CaL)
        # 10: Xr1   (activation in i_Kr)
        # 11: Xr2  (inactivation in i_Kr
        # 12: Xs   (activation in i_Ks)
        # 13: h    (inactivation in i_Na)
        # 14: j    (slow inactivation in i_Na)
        # 15: m    (activation in i_Na)
        # 16: Xf   (inactivation in i_f)
        # 17: s    (inactivation in i_to)
        # 18: r    (activation in i_to)
        # 19: dCaT (activation in i_CaT)
        # 20: fCaT (inactivation in i_CaT)
        # 21: R (in Irel)
        # 22: O (in Irel)
        # 23: I (in Irel)
        """

        self.y_voltage.append(y[0])
        self.t.append(t)
        self.full_y.append(y)

        d_y = np.zeros(23)

        # --------------------------------------------------------------------
        # Reversal Potentials:
        E_Ca = 0.5 * self.R * self.T / self.F * log(self.Cao / y[2])  # millivolt
        E_Na = self.R * self.T / self.F * log(self.Nao / y[3])  # millivolt
        E_K = self.R * self.T / self.F * log(self.Ko / y[4])  # millivolt

        # --------------------------------------------------------------------
        # Inward Rectifier K+ current (Ik1):
        # define parameters from x_K1
        xK11 = self.x_K1[1]
        xK12 = self.x_K1[2]
        xK13 = self.x_K1[3]
        xK14 = self.x_K1[4]
        xK15 = self.x_K1[5]

        alpha_xK1 = xK11*exp((y[0]+xK13)/xK12)
        beta_xK1 = exp((y[0]+xK15)/xK14)
        XK1_inf = alpha_xK1/(alpha_xK1+beta_xK1)

        # Current:
        g_K1 = self.x_K1[0] * self.default_parameters['G_K1']
        i_K1 = g_K1*XK1_inf*(y[0]-E_K)*sqrt(self.Ko/5.4)

        # Rapid Delayed Rectifier Current (Ikr):
        # define parameters from x_KR
        Xr1_1 = self.x_KR[1]
        Xr1_2 = self.x_KR[2]
        Xr1_5 = self.x_KR[3]
        Xr1_6 = self.x_KR[4]
        Xr2_1 = self.x_KR[5]
        Xr2_2 = self.x_KR[6]
        Xr2_5 = self.x_KR[7]
        Xr2_6 = self.x_KR[8]

        # parameter-dependent values:
        Xr1_3 = Xr1_5*Xr1_1
        Xr2_3 = Xr2_5*Xr2_1
        Xr1_4 = 1/((1/Xr1_2)+(1/Xr1_6))
        Xr2_4 = 1/((1/Xr2_2)+(1/Xr2_6))

        # 10: Xr1 (dimensionless) (activation in i_Kr_Xr1)
        alpha_Xr1 = Xr1_1*exp((y[0])/Xr1_2)
        beta_Xr1 = Xr1_3*exp((y[0])/Xr1_4)
        Xr1_inf = alpha_Xr1/(alpha_Xr1 + beta_Xr1)
        tau_Xr1 = ((1./(alpha_Xr1 + beta_Xr1))+self.x_KR[9])
        d_y[9] = (Xr1_inf-y[9])/tau_Xr1

        # 11: Xr2 (dimensionless) (inactivation in i_Kr_Xr2)
        alpha_Xr2 = Xr2_1*exp((y[0])/Xr2_2)
        beta_Xr2 = Xr2_3*exp((y[0])/Xr2_4)
        Xr2_inf = alpha_Xr2/(alpha_Xr2+beta_Xr2)
        tau_Xr2 = ((1./(alpha_Xr2+beta_Xr2))+self.x_KR[10])
        d_y[10] = (Xr2_inf-y[10])/tau_Xr2

        # Current:
        g_Kr = self.x_KR[0]*self.default_parameters['G_Kr']  # nS_per_pF (in i_Kr)
        i_Kr = g_Kr*(y[0]-E_K)*y[9]*y[10]*sqrt(self.Ko/5.4)

        # Slow delayed rectifier current (IKs):
        # define parameters from x_IKS:
        ks1 = self.x_IKS[1]
        ks2 = self.x_IKS[2]
        ks5 = self.x_IKS[3]
        ks6 = self.x_IKS[4]
        tauks_const = self.x_IKS[5]

        # parameter-dependent values:
        ks3 = ks5*ks1
        ks4 = 1/((1/ks2)+(1/ks6))

        # 12: Xs (dimensionless) (activation in i_Ks)
        alpha_Xs = ks1*exp((y[0])/ks2)
        beta_Xs = ks3*exp((y[0])/ks4)
        Xs_inf = alpha_Xs/(alpha_Xs+beta_Xs)
        tau_Xs = (1./(alpha_Xs+beta_Xs)) + tauks_const
        d_y[11] = (Xs_inf-y[11])/tau_Xs

        # Current:
        g_Ks = self.x_IKS[0]*self.default_parameters['G_Ks']    # nS_per_pF (in i_Ks)
        i_Ks = g_Ks*(y[0]-E_K)*(y[11]**2)

        # Transient outward current (Ito):
        # define parameters from xTO
        r1 = self.xTO[1]
        r2 = self.xTO[2]
        r5 = self.xTO[3]
        r6 = self.xTO[4]
        s1 = self.xTO[5]
        s2 = self.xTO[6]
        s5 = self.xTO[7]
        s6 = self.xTO[8]
        tau_r_const = self.xTO[9]
        tau_s_const = self.xTO[10]

        # parameter-dependent values:
        r3 = r5*r1
        r4 = 1/((1/r2)+(1/r6))
        s3 = s5*s1
        s4 = 1/((1/s2)+(1/s6))

        # 17: s (dimensionless) (inactivation in i_to)
        alpha_s = s1*exp((y[0])/s2)
        beta_s = s3*exp((y[0])/s4)
        s_inf = alpha_s/(alpha_s+beta_s)
        tau_s = ((1./(alpha_s+beta_s))+tau_s_const)
        d_y[16] = (s_inf-y[16])/tau_s

        # 18: r (dimensionless) (activation in i_to)
        alpha_r = r1*exp((y[0])/r2)
        beta_r = r3*exp((y[0])/r4)
        r_inf = alpha_r/(alpha_r + beta_r)
        tau_r = (1./(alpha_r + beta_r))+tau_r_const
        d_y[17] = (r_inf-y[17])/tau_r

        # Current:
        g_to = self.xTO[0]*self.default_parameters['G_to']  # nS_per_pF (in i_to)
        i_to = g_to*(y[0]-E_K)*y[16]*y[17]

        # L-type Ca2+ current (ICaL):
        # define parameters from x_cal
        d1 = self.x_cal[1]
        d2 = self.x_cal[2]
        d5 = self.x_cal[3]
        d6 = self.x_cal[4]
        f1 = self.x_cal[5]
        f2 = self.x_cal[6]
        f5 = self.x_cal[7]
        f6 = self.x_cal[8]
        taud_const = self.x_cal[9]
        tauf_const = self.x_cal[10]

        # parameter-dependent values:
        d3 = d5*d1
        d4 = 1/((1/d2)+(1/d6))
        f3 = f5*f1
        f4 = 1/((1/f2)+(1/f6))

        # 7: d (dimensionless) (activation in i_CaL)
        alpha_d = d1*exp(((y[0]))/d2)
        beta_d = d3*exp(((y[0]))/d4)
        d_inf = alpha_d/(alpha_d + beta_d)
        tau_d = ((1/(alpha_d + beta_d))+taud_const)
        d_y[6] = (d_inf-y[6])/tau_d

        # 8: f (dimensionless) (inactivation  i_CaL)
        alpha_f = f1*exp(((y[0]))/f2)
        beta_f = f3*exp(((y[0]))/f4)
        f_inf = alpha_f/(alpha_f+beta_f)
        tau_f = ((1./(alpha_f+beta_f)) + tauf_const)
        d_y[7] = (f_inf-y[7])/tau_f

        # 9: fCa (dimensionless) (calcium-dependent inactivation in i_CaL)
        # from Ten tusscher 2004
        scale_Ical_Fca_Cadep = 1.2
        alpha_fCa = 1.0/(1.0+((scale_Ical_Fca_Cadep*y[2])/.000325) ** 8.0)
        beta_fCa = 0.1/(1.0+exp((scale_Ical_Fca_Cadep*y[2]-.0005)/0.0001))
        gamma_fCa = .2/(1.0+exp((scale_Ical_Fca_Cadep*y[2]-0.00075)/0.0008))

        fCa_inf = ((alpha_fCa+beta_fCa+gamma_fCa+.23)/(1.46))
        tau_fCa = 2  # ms
        if ((fCa_inf > y[8]) and (y[0] > -60)):
            k_fca = 0
        else:
            k_fca = 1

        d_y[8] = k_fca*(fCa_inf-y[8])/tau_fCa

        # Current:
        p_CaL = self.x_cal[0]*self.default_parameters['P_CaL']  # nS_per_pF (in i_CaL)
        p_CaL_shannonCa = 5.4e-4
        p_CaL_shannonNa = 1.5e-8
        p_CaL_shannonK = 2.7e-7
        p_CaL_shannonTot = p_CaL_shannonCa + p_CaL_shannonNa + p_CaL_shannonK
        p_CaL_shannonCap = p_CaL_shannonCa/p_CaL_shannonTot
        p_CaL_shannonNap = p_CaL_shannonNa/p_CaL_shannonTot
        p_CaL_shannonKp = p_CaL_shannonK/p_CaL_shannonTot

        p_CaL_Ca = p_CaL_shannonCap*p_CaL
        p_CaL_Na = p_CaL_shannonNap*p_CaL
        p_CaL_K = p_CaL_shannonKp*p_CaL

        ibarca = p_CaL_Ca*4.0*y[0]*self.F ** 2.0/(self.R*self.T) * (.341*y[2]*exp(
            2.0*y[0]*self.F/(self.R*self.T))-0.341*self.Cao)/(exp(2.0*y[0]*self.F/(self.R*self.T))-1.0)
        i_CaL_Ca = ibarca * y[6]*y[7]*y[8]

        ibarna = p_CaL_Na * \
            y[0]*self.F ** 2.0/(self.R*self.T) * (.75*y[3]*exp(y[0]*self.F/(self.R*self.T)) -
                                  0.75*self.Nao)/(exp(y[0]*self.F/(self.R*self.T))-1.0)
        i_CaL_Na = ibarna * y[6]*y[7]*y[8]

        ibark = p_CaL_K*y[0]*self.F ** 2.0/(self.R*self.T) * (.75*y[4] *
                                              exp(y[0]*self.F/(self.R*self.T))-0.75*self.Ko)/(exp(
                                                  y[0]*self.F/(self.R*self.T))-1.0)
        i_CaL_K = ibark * y[6]*y[7]*y[8]

        i_CaL = i_CaL_Ca+i_CaL_Na+i_CaL_K

        # T-type Calcium Current (ICaT):
        # SAN T-TYPE CA2+ model (Demir et al., Maltsev-Lakatta ),
        # G_CaT determined by fit to Kurokawa IV:

        # 19: dCaT (activation in i_CaT)
        dcat_inf = 1./(1+exp(-((y[0]) + 26.3)/6))
        tau_dcat = 1./(1.068*exp(((y[0])+26.3)/30) + 1.068*exp(-((y[0])+26.3)/30))
        d_y[18] = (dcat_inf-y[18])/tau_dcat

        # 20: fCaT (inactivation in i_CaT)
        fcat_inf = 1./(1+exp(((y[0]) + 61.7)/5.6))
        tau_fcat = 1./(.0153*exp(-((y[0])+61.7)/83.3) + 0.015*exp(
            ((y[0])+61.7)/15.38))
        d_y[19] = (fcat_inf-y[19])/tau_fcat

        g_CaT = self.x_cat*self.default_parameters['G_CaT']  # nS_per_pF (in i_CaT)
        i_CaT = g_CaT*(y[0]-E_Ca)*y[18]*y[19]

        # Sodium Current (INa):
        # define parameters from x_Na
        m1 = self.x_NA[1]
        m2 = self.x_NA[2]
        m5 = self.x_NA[3]
        m6 = self.x_NA[4]
        h1 = self.x_NA[5]
        h2 = self.x_NA[6]
        h5 = self.x_NA[7]
        h6 = self.x_NA[8]
        j1 = self.x_NA[9]
        j2 = self.x_NA[10]
        tau_m_const = self.x_NA[11]
        tau_h_const = self.x_NA[12]
        tau_j_const = self.x_NA[13]

        # parameter-dependent values:
        m3 = m5*m1
        m4 = 1/((1/m2)+(1/m6))
        h3 = h5*h1
        h4 = 1/((1/h2)+(1/h6))
        j5 = h5
        j6 = h6
        j3 = j5*j1
        j4 = 1/((1/j2)+(1/j6))

        # 13: h (dimensionless) (inactivation in i_Na)
        alpha_h = h1*exp((y[0])/h2)
        beta_h = h3*exp((y[0])/h4)
        h_inf = (alpha_h/(alpha_h+beta_h))
        tau_h = ((1./(alpha_h+beta_h))+tau_h_const)
        d_y[12] = (h_inf-y[12])/tau_h

        # 14: j (dimensionless) (slow inactivation in i_Na)
        alpha_j = j1*exp((y[0])/j2)
        beta_j = j3*exp((y[0])/j4)
        j_inf = (alpha_j/(alpha_j+beta_j))
        tau_j = ((1./(alpha_j+beta_j))+tau_j_const)
        d_y[13] = (j_inf-y[13])/tau_j

        # 15: m (dimensionless) (activation in i_Na)
        alpha_m = m1*exp((y[0])/m2)
        beta_m = m3*exp((y[0])/m4)
        m_inf = alpha_m/(alpha_m+beta_m)
        tau_m = ((1./(alpha_m+beta_m))+tau_m_const)
        d_y[14] = (m_inf-y[14])/tau_m

        # Current:
        g_Na = self.x_NA[0]*self.default_parameters['G_Na']  # nS_per_pF (in i_Na)
        i_Na = g_Na*y[14] ** 3.0*y[12]*y[13]*(y[0]-E_Na)

        # Funny/HCN current (If):
        # define parameters from x_F
        xF1 = self.x_F[1]
        xF2 = self.x_F[2]
        xF5 = self.x_F[3]
        xF6 = self.x_F[4]
        xF_const = self.x_F[5]

        # parameter-dependent values:
        xF3 = xF5*xF1
        xF4 = 1/((1/xF2)+(1/xF6))

        # 16: Xf (dimensionless) (inactivation in i_f)
        alpha_Xf = xF1*exp((y[0])/xF2)
        beta_Xf = xF3*exp((y[0])/xF4)
        Xf_inf = alpha_Xf/(alpha_Xf+beta_Xf)
        tau_Xf = ((1./(alpha_Xf+beta_Xf))+xF_const)
        d_y[15] = (Xf_inf-y[15])/tau_Xf

        # Current:
        g_f = self.x_F[0]*self.default_parameters['G_F']  # nS_per_pF (in i_f)
        NatoK_ratio = .491  # Verkerk et al. 2013
        Na_frac = NatoK_ratio/(NatoK_ratio+1)
        i_fNa = Na_frac*g_f*y[15]*(y[0]-E_Na)
        i_fK = (1-Na_frac)*g_f*y[15]*(y[0]-E_K)
        i_f = i_fNa+i_fK

        # Na+/Ca2+ Exchanger current (INaCa):
        # Ten Tusscher formulation
        KmCa = 1.38    # Cai half-saturation constant millimolar (in i_NaCa)
        KmNai = 87.5    # Nai half-saturation constnat millimolar (in i_NaCa)
        Ksat = 0.1    # saturation factor dimensionless (in i_NaCa)
        gamma = 0.35*2    # voltage dependence parameter dimensionless (in i_NaCa)
        # factor to enhance outward nature of inaca dimensionless (in i_NaCa)
        alpha = 2.5*1.1
        # maximal inaca pA_per_pF (in i_NaCa)
        kNaCa = 1000*1.1*self.default_parameters['K_NaCa']

        i_NaCa = kNaCa*((exp(gamma*y[0]*self.F/(self.R*self.T))*(y[3] ** 3.0)*self.Cao)-(exp(
            (gamma-1.0)*y[0]*self.F/(self.R*self.T))*(
            self.Nao ** 3.0)*y[2]*alpha))/(((KmNai ** 3.0)+(self.Nao ** 3.0))*(KmCa+self.Cao)*(
                1.0+Ksat*exp((gamma-1.0)*y[0]*self.F/(self.R*self.T))))

        # Na+/K+ pump current (INaK):
        # Ten Tusscher formulation
        Km_K = 1.0    # Ko half-saturation constant millimolar (in i_NaK)
        Km_Na = 40.0  # Nai half-saturation constant millimolar (in i_NaK)
        # maxiaml nak pA_per_pF (in i_NaK)
        PNaK = 1.362*1.818*self.default_parameters['P_NaK']
        i_NaK = PNaK*((self.Ko*y[3])/((self.Ko+Km_K)*(y[3]+Km_Na)*(1.0 + 0.1245*exp(
            -0.1*y[0]*self.F/(self.R*self.T))+0.0353*exp(-y[0]*self.F/(self.R*self.T)))))

        # SR Uptake/SERCA (J_up):
        # Ten Tusscher formulation
        Kup = 0.00025*0.702    # millimolar (in calcium_dynamics)
        # millimolar_per_milisecond (in calcium_dynamics)
        VmaxUp = 0.000425 * 0.26 * self.default_parameters['VmaxUp']
        i_up = VmaxUp/(1.0+Kup ** 2.0/y[2] ** 2.0)

        # SR Leak (J_leak):
        # Ten Tusscher formulation
        V_leak = self.default_parameters['V_leak']*0.00008*0.02
        i_leak = (y[1]-y[2])*V_leak

        # SR Release/RYR (J_rel):
        # re-fit parameters. scaled to account for differences in calcium
        # concentration in cleft (cleft is used in shannon-bers model geometry,
        # not in this model geometry)
        ks = 12.5*self.default_parameters['ks']  # [1/ms]
        koCa = 56320*11.43025              # [mM**-2 1/ms]
        kiCa = 54*0.3425                   # [1/mM/ms]
        kom = 1.5*0.1429                   # [1/ms]
        kim = 0.001*0.5571                 # [1/ms]
        ec50SR = 0.45
        MaxSR = 15
        MinSR = 1

        kCaSR = MaxSR - (MaxSR-MinSR)/(1+(ec50SR/y[1])**2.5)
        koSRCa = koCa/kCaSR
        kiSRCa = kiCa*kCaSR
        RI = 1-y[20]-y[21]-y[22]

        d_y[20] = (kim*RI-kiSRCa*y[2]*y[20]) - (koSRCa*y[2]**2*y[20]-kom*y[21])
        d_y[21] = (koSRCa*y[2]**2*y[20]-kom*y[21]) - (kiSRCa*y[2]*y[21]-kim*y[22])
        d_y[22] = (kiSRCa*y[2]*y[21]-kim*y[22]) - (kom*y[22]-koSRCa*y[2]**2*RI)

        i_rel = ks*y[21]*(y[1]-y[2])*(self.V_SR/self.Vc)

        # Background Sodium (I_bNa):
        # Ten Tusscher formulation
        g_b_Na = .00029*1.5*self.default_parameters['G_b_Na']    # nS_per_pF (in i_b_Na)
        i_b_Na = g_b_Na*(y[0]-E_Na)

        # Background Calcium (I_bCa):
        # Ten Tusscher formulation
        g_b_Ca = .000592*0.62*self.default_parameters['G_b_Ca']    # nS_per_pF (in i_b_Ca)
        i_b_Ca = g_b_Ca*(y[0]-E_Ca)

        # Calcium SL Pump (I_pCa):
        # Ten Tusscher formulation
        g_PCa = 0.025*10.5*self.default_parameters['G_PCa']    # pA_per_pF (in i_PCa)
        KPCa = 0.0005    # millimolar (in i_PCa)
        i_PCa = g_PCa*y[2]/(y[2]+KPCa)

        # 2: CaSR (millimolar)
        # rapid equilibrium approximation equations --
        # not as formulated in ten Tusscher 2004 text
        Buf_SR = 10.0*1.2  # millimolar (in calcium_dynamics)
        Kbuf_SR = 0.3  # millimolar (in calcium_dynamics)
        Ca_SR_bufSR = 1/(1.0+Buf_SR*Kbuf_SR/(y[1]+Kbuf_SR)**2.0)

        d_y[1] = Ca_SR_bufSR*self.Vc/self.V_SR*(i_up-(i_rel+i_leak))

        # 3: Cai (millimolar)
        # rapid equilibrium approximation equations --
        # not as formulated in ten Tusscher 2004 text
        Buf_C = .06  # millimolar (in calcium_dynamics)
        Kbuf_C = .0006  # millimolar (in calcium_dynamics)
        Cai_bufc = 1/(1.0+Buf_C*Kbuf_C/(y[2]+Kbuf_C)**2.0)

        d_y[2] = (Cai_bufc)*(i_leak-i_up+i_rel-d_y[5] - 
                (i_CaL_Ca+i_CaT+i_b_Ca+i_PCa-2*i_NaCa)*self.Cm/(2.0*self.Vc*self.F))

        # 4: Nai (millimolar) (in sodium_dynamics)
        d_y[3] = -self.Cm*(i_Na+i_b_Na+i_fNa+3.0*i_NaK+3.0*i_NaCa + i_CaL_Na)/(self.F*self.Vc)

        # 5: Ki (millimolar) (in potatssium_dynamics)
        d_y[4] = -self.Cm*(i_K1+i_to+i_Kr+i_Ks+i_fK - 2.*i_NaK + i_CaL_K)/(self.F*self.Vc)

        i_no_ion = 0
        if self.is_no_ion_selective:
            current_dictionary = {
                'I_K1':    i_K1,
                'I_To':    i_to,
                'I_Kr':    i_Kr,
                'I_Ks':    i_Ks,
                'I_CaL':   i_CaL_Ca,
                'I_NaK':   i_NaK,
                'I_Na':    i_Na,
                'I_NaCa':  i_NaCa,
                'I_pCa':   i_PCa,
                'I_F':     i_f,
                'I_bNa':   i_b_Na,
                'I_bCa':   i_b_Ca,
                'I_CaT':   i_CaT,
                'I_up':    i_up,
                'I_leak':  i_leak
            }
            for curr_name, scale in self.no_ion_selective.items():
                i_no_ion += scale * current_dictionary[curr_name]

        d_y[0] = -(i_K1+i_to+i_Kr+i_Ks+i_CaL+i_CaT+i_NaK+i_Na+i_NaCa +
                   i_PCa+i_f+i_b_Na+i_b_Ca + i_no_ion)

        if self.current_response_info:
            current_timestep = [
                trace.Current(name='I_K1', value=i_K1),
                trace.Current(name='I_To', value=i_to),
                trace.Current(name='I_Kr', value=i_Kr),
                trace.Current(name='I_Ks', value=i_Ks),
                trace.Current(name='I_CaL', value=i_CaL_Ca),
                trace.Current(name='I_NaK', value=i_NaK),
                trace.Current(name='I_Na', value=i_Na),
                trace.Current(name='I_NaCa', value=i_NaCa),
                trace.Current(name='I_pCa', value=i_PCa),
                trace.Current(name='I_F', value=i_f),
                trace.Current(name='I_bNa', value=i_b_Na),
                trace.Current(name='I_bCa', value=i_b_Ca),
                trace.Current(name='I_CaT', value=i_CaT),
                trace.Current(name='I_up', value=i_up),
                trace.Current(name='I_leak', value=i_leak)
            ]
            self.current_response_info.currents.append(current_timestep)

        self.d_y_voltage.append(d_y[0])

        return d_y


def generate_trace(protocol: protocols.PROTOCOL_TYPE,
                   tunable_parameters: List[ga_configs.Parameter] = None,
                   params: List[float] = None) -> trace.Trace:
    """Generates a trace.

    Leave `params` argument empty if generating baseline trace with
    default parameter values.

    Args:
        tunable_parameters: List of tunable parameters.
        protocol: A protocol object used to generate the trace.
        params: A set of parameter values (where order must match with ordered
            labels in `tunable_parameters`).

    Returns:
        A Trace object.
    """
    new_params = dict()
    if params and tunable_parameters:
        for i in range(len(params)):
            new_params[tunable_parameters[i].name] = params[i]

    return KernikModel(updated_parameters=new_params).generate_response(protocol)
