
import numpy as np

def contingency_table(label1, label2, k1, k2):
    #contingency table
    cont_Table = np.zeros((k1, k2),dtype=int)

    for i in range(len(label1)):
        cont_Table[label1[i],label2[i]]+=1

    return cont_Table


def ARI(label1, label2):

    k1 = np.amax(label1)+1
    k2 = np.amax(label2)+1

    cont_Table = contingency_table(label1, label2, k1, k2)

    sum1= np.sum(cont_Table,axis=1)
    sum2 = np.sum(cont_Table, axis=0)

    nij = 0
    ai2 = 0
    bi2 = 0

    n2 = len(label1)*(len(label1)-1)/2

    for i in range(k1):
        for j in range(k2):
            nij += cont_Table[i][j]*(cont_Table[i][j]-1)/2

    for i in range(k1):
        ai2 += sum1[i]*(sum1[i]-1)/2

    for i in range(k2):
        bi2 += sum2[i] * (sum2[i] - 1) / 2

    ari = nij-(ai2*bi2/n2)
    ari /= ((ai2+bi2)/2) - (ai2*bi2/n2)

    return ari


# main function
if __name__ == '__main__':

    label1= [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    label2 = [0, 1, 0, 1, 1, 2, 2, 2, 2, 2]

    ari = ARI(label1, label2)

    print('Adjusted Rand index = ',ari)