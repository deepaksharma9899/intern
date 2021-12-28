
import numpy as np

def contingency_table(label1, label2, k1, k2):
    #contingency table
    cont_Table = np.zeros((k1, k2),dtype=int)

    for i in range(len(label1)):
        cont_Table[label1[i],label2[i]]+=1

    return cont_Table


def Fowlkes_mallows(label1, label2):

    k1 = np.amax(label1)+1
    k2 = np.amax(label2)+1
    n=len(label1)

    cont_Table = contingency_table(label1, label2, k1, k2)

    sum1= np.sum(cont_Table,axis=1)
    sum2 = np.sum(cont_Table, axis=0)

    Tk = 0
    Pk = 0
    Qk = 0

    for i in range(k1):
        for j in range(k2):
            Tk += cont_Table[i][j]*cont_Table[i][j]

    for i in range(k1):
        Pk += sum1[i]* sum1[i]

    for i in range(k2):
        Qk += sum2[i] * sum2[i]

    Tk -= n
    Pk -= n
    Qk -= n

    #corner case
    if Pk==0 or Qk==0:
        return 0

    fms = Tk/(np.sqrt(Pk*Qk))

    return fms


# main function
if __name__ == '__main__':

    label1 = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    label2 = [0, 1, 0, 1, 1, 2, 2, 2, 2, 2]

    fms = Fowlkes_mallows(label1, label2)

    print('Fowlkes mallows Score = ',fms)