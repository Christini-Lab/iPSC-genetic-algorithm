

def jacobianWrapper(time, Y):
    tDrugApplication = 10000
    INaFRedMed = 1
    ICaLRedMed = 1
    IKrRedMed = 1
    IKsRedMed = 1
    
    return paciJacobian(time,y, tDrugApplication, INaFRedMed,
             ICaLRedMed, IKrRedMed, IKsRedMed)


def paciJacobian(time, Y, tDrugApplication, INaFRedMed, ICaLRedMed, IKrRedMed, IKsRedMed):
    # dY is dY/dt
    dY_dt = Paci2018(time, Y, tDrugApplication, INaFRedMed,ICaLRedMed, IKrRedMed, IKsRedMed)
    
    # find perturbations
    Y_step = Y + Y * .001

    n=len(Y)
    jac=np.zeros((n, n))
    
    for column in range(n): #through columns
        Y_step_current = np.copy(Y)
        Y_step_current[column] = Y_step[column]
        
        
        dY_dY = Paci2018(time, Y_step_current, tDrugApplication, INaFRedMed, ICaLRedMed, IKrRedMed, IKsRedMed)

        if (Y_step[column] - Y[column]) != 0:
            jac[column, :] = (dY_dY - dY_dt) / (Y_step[column] - Y[column])

    return jac

