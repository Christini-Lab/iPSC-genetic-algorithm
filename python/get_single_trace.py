import paci_2018
import protocols
import pdb
import matplotlib.pyplot as plt

VC_PROTOCOL_NEGATIVE80 = protocols.VoltageClampProtocol(
    steps=[
        protocols.VoltageClampStep(duration=5.0, voltage=-0.08),
    ]
)


def main():
    """Run parameter tuning or voltage clamp protocol experiments here
    """
    paci_model = paci_2018.PaciModel()
    paci_model.generate_response(protocol = VC_PROTOCOL_NEGATIVE80)
    paci_trace = paci_2018.generate_trace(protocol = VC_PROTOCOL_NEGATIVE80)
    fig = plt.figure(figsize=(10, 5))

    ax_1 = fig.add_subplot(511)
    ax_1.plot(
            [1000 * i for i in paci_trace.t],
            [i * 1000 for i in paci_trace.y],
            'b',
            label='Voltage')
    plt.xlabel('Time (ms)')
    plt.ylabel(r'$V_m$ (mV)')

    I_NaCa = []
    I_CaL = []
    Cai = []
    CaSR = []
    i = 0
    for currents in paci_trace.current_response_info.currents:
        I_CaL.append(currents[4].value)
        I_NaCa.append(currents[8].value)
        Cai.append(paci_model.full_y[i][2])
        CaSR.append(paci_model.full_y[i][1])
        i = i + 1

    ax_2 = fig.add_subplot(512) 
    ax_2.plot(
        [1000 * i for i in paci_trace.t],
        I_CaL,
        'r--',
            label='I_CaL')
    plt.ylabel(r'$I_{CaL}$ (nA/nF)')


    ax_3 = fig.add_subplot(513) 
    ax_3.plot(
        [1000 * i for i in paci_trace.t],
        I_NaCa,
        'r--',
            label='I_NaCa')
    plt.ylabel(r'$I_{NaCa}$ (nA/nF)')
    
    ax_4 = fig.add_subplot(514) 
    ax_4.plot(
        [1000 * i for i in paci_trace.t],
        Cai,
        'r--',
            label='Ca_i')
    plt.ylabel(r'$Cai$ (nA/nF)')

    ax_5 = fig.add_subplot(515) 
    ax_5.plot(
        [1000 * i for i in paci_trace.t],
        CaSR,
        'r--',
            label='CaSR')
    plt.ylabel(r'$Ca_{SR}$ (nA/nF)')
    
    plt.show()

    pdb.set_trace()


if __name__ == '__main__':
    main()
