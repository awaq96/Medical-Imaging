import numpy as np

def idealLowpassFilter(emptymask, cutoff):
    mask = np.zeros(emptymask)
    for u in range(0, emptymask[0]):
        for v in range(0, emptymask[1]):
            M = pow((u - (emptymask[0] / 2)), 2)
            N = pow((v - (emptymask[1] / 2)), 2)

            Duv = np.sqrt(M + N)
            if Duv <= cutoff:
                mask[u, v] = 1
            else:
                mask[u, v] = 0

    return mask


def idealHighpassFilter(emptymask, cutoff):

    mask = np.zeros(emptymask)
    for u in range(0, emptymask[0]):
        for v in range(0, emptymask[1]):
            M = pow((u - (emptymask[0] / 2)), 2)
            N = pow((v - (emptymask[1] / 2)), 2)

            Duv = np.sqrt(M + N)
            if Duv <= cutoff:
                mask[u, v] = 0
            else:
                mask[u, v] = 1
    return mask

def gaussianLowpassFilter(emptymask, cutoff):

    lowpassG = np.zeros(emptymask)

    for u in range(0, emptymask[0]):
        for v in range(0, emptymask[1]):
            P = pow((u - (emptymask[0] / 2)), 2)
            Q = pow((v - (emptymask[1] / 2)), 2)

            Duv = np.sqrt(P + Q)

            exponent = -1 * pow(Duv, 2) / (2 * pow(cutoff, 2))
            lowpassG[u, v] = pow(np.e, exponent)

    return lowpassG


def gaussianHighpassFilter(emptymask, cutoff):
    lowpass = gaussianLowpassFilter(emptymask, cutoff)
    highpass = np.zeros(emptymask)

    for u in range(emptymask[0]):
        for v in range(emptymask[1]):
            highpass[u, v] = 1 - lowpass[u, v]
    return highpass


def butterworthLowpassFilter(emptymask, cutoff, order):

    blpf = np.zeros(emptymask)

    for u in range(0, emptymask[0]):
        for v in range(0, emptymask[1]):
            P = pow((u - (emptymask[0] / 2)), 2)
            Q = pow((v - (emptymask[1] / 2)), 2)

            Duv = np.sqrt(P + Q)

            blpf[u, v] = 1 / (1 + pow((Duv / cutoff), 2 * order))

    return blpf


def butterworthHighpassFilter(emptymask, cutoff, order):
    lowpass = butterworthLowpassFilter(emptymask, cutoff, order)
    highpass = np.zeros(emptymask)

    for u in range(emptymask[0]):
        for v in range(emptymask[1]):
            highpass[u, v] = 1 - lowpass[u, v]

    return highpass


def ringLowpassFilter(emptymask, cutoff, thickness):
    mask = np.zeros(emptymask)
    ring_end = cutoff - thickness
    for u in range(0, emptymask[0]):
        for v in range(0, emptymask[1]):
            M = pow((u - (emptymask[0] / 2)), 2)
            N = pow((v - (emptymask[1] / 2)), 2)

            Duv = np.sqrt(M + N)
            if ring_end<= Duv  and Duv <= cutoff:
                mask[u, v] = 1
            else:
                mask[u, v] = 0
    return mask


def ringHighpassFilter(emptymask, cutoff, thickness):
    lowpass = ringLowpassFilter(emptymask, cutoff, thickness)
    highpass = np.zeros(emptymask)

    for u in range(emptymask[0]):
        for v in range(emptymask[1]):
            highpass[u, v] = 1 - lowpass[u, v]
    return highpass

