"""
The purpose of this file is to teach someone about the protocols that are
present in this directory, how to use them, and how to add new ones.

At the time of writing this (11/26/19), there are three different types of
protocols. Their behaviors are described in the following three classes:
    - SingleActionPotentialProtocol
    - IrregularPacingProtocol
    - VoltageClampProtocol

The rationale for creating these protocols as their own class is so you can 
create multiple instances with different states, then apply them to different
computational models.

You can find these protocols in ./protocols.py. Let's take a look at how import
and run each w/ the Kernik Model:
"""


##Single Action Potential
"""
The SingleActionPotentialProtocol() class name is a misnomer. It should
be called SpontaneousProtocol(). You initialize the protocol with:
    - duration: float – the amount of time you want the model to run
                spontaneously
"""
# 1. Imports 
from kernik import KernikModel
import protocols
import matplotlib.pyplot as plt

DURATION = 983  # Length of Kernik AP, in milliseconds
SAP_PROTOCOL_KERNIK = protocols.SingleActionPotentialProtocol(DURATION)

# STOP AND OPEN ./protocols.py. Look at the SingleActionPotentialProtocol
# so you understand what the kernik_protocol variable holds.

baseline_kernik = KernikModel() # Initialize the baseline kernik individual
baseline_kernik.generate_response(SAP_PROTOCOL_KERNIK) # Run model

plt.plot(baseline_kernik.t, baseline_kernik.y_voltage)
plt.show()


## ----------------------------------------------------------------------


##VC Protocol
VC_PROTOCOL = protocols.VoltageClampProtocol(
    steps=[
        protocols.VoltageClampStep(duration=0.050, voltage=-0.08),
        protocols.VoltageClampStep(duration=0.050, voltage=-0.12),
        protocols.VoltageClampStep(duration=0.500, voltage=-0.057),
        protocols.VoltageClampStep(duration=0.025, voltage=-0.04),
        protocols.VoltageClampStep(duration=0.075, voltage=0.02),
        protocols.VoltageClampStep(duration=0.025, voltage=-0.08),
        protocols.VoltageClampStep(duration=0.250, voltage=0.04),
        protocols.VoltageClampStep(duration=1.900, voltage=-0.03),
        protocols.VoltageClampStep(duration=0.750, voltage=0.04),
        protocols.VoltageClampStep(duration=1.725, voltage=-0.03),
        protocols.VoltageClampStep(duration=0.650, voltage=-0.08),
    ]
)



#def plot_voltage_currents():
#    kernik_protocol = protocols.SingleActionPotentialProtocol(983)
#    baseline_kernik = KernikModel()
#    baseline_kernik.generate_response(kernik_protocol)
#
#    coi = ['I_K1', 'I_Kr', 'I_Ks', 'I_CaL', 'I_Na', 'I_To']
#    currents_df = get_current_data(baseline_kernik.current_response_info.currents, coi, baseline_kernik.t)
#
#    fig = plt.figure(figsize=(10,8))
#    plt.plot(baseline_kernik.t, baseline_kernik.y_voltage, label='Kernik')
#    plt.axis([0, 983, -80, 40])
#    ax = plt.gca()
#    ax.tick_params(labelsize=18)
#    ax.tick_params(axis='x', labelcolor='none')
#    plt.ylabel('Voltage (mV)', fontsize=20)
#    #plt.savefig('figures/voltage.svg')
#
#    plt.show()
#
#def get_current_data(currents, currents_of_interest, time):
#    new_currents = []
#    for temp_currents in currents:
#        currents_of_interest
#        new_currents.append([current.value for current in temp_currents if current.name in ['I_K1', 'I_Kr', 'I_Ks', 'I_CaL', 'I_Na', 'I_To']])
#    
#    new_currents = np.array(new_currents)
#    currents_df = pd.DataFrame(new_currents, columns=currents_of_interest)
#    currents_df['Time (ms)'] = np.array(time)
#    fig = plt.figure(figsize=(10,8))
#    axarr = fig.add_subplot(611)
#    axes = currents_df.set_index('Time (ms)').plot(subplots=True, figsize=(8,6), ax=axarr, sharex=True, fontsize=18)
#    plt.xticks(rotation=0)
#    axes[5].set_xlabel('Time (ms)', fontsize=20)
#    index = 0
#    for ax in axes:
#        ax.legend(loc=2)
#        #ax.tick_params(axis='y', length=0, labelcolor='none')
#
#    ax = fig.add_subplot(111, frameon=False)
#    ax.tick_params(labelcolor='none', length=0, top='off', bottom='off', left='off', right='off')
#    ax.set_ylabel('Currents', labelpad=20, fontsize=20)
#    #plt.savefig('figures/currents.svg')
#    return currents_df
#
#plot_voltage_currents()
#
#
#
##paci_protocol = protocols.SingleActionPotentialProtocol(1.8)
##baseline_paci = PaciModel()
##baseline_paci.generate_response(paci_protocol)
#
#
##t_max_p = baseline_paci.t[np.array(baseline_paci.y_voltage).argmax()]*1000
##t_max_k = baseline_kernik.t[np.array(baseline_kernik.y_voltage).argmax()]
#pdb.set_trace()
#
##plt.plot(np.array(baseline_paci.t)*1000-t_max_p, [v*1000 for v in baseline_paci.y_voltage], label='Paci')
##plt.plot(np.array(baseline_kernik.t)-t_max_k, baseline_kernik.y_voltage, label='Kernik')
##plt.xlabel('Time (ms)', fontsize=20)
##plt.ylabel('Voltage', fontsize=20)
##plt.legend()
##plt.show()

