# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# utility function
def util(c, utype, params):
    if utype == 'CRRA':
        [gamma] = params
        u = (c**(1-gamma) - 1.)/(1-gamma)
        mu = c**(-gamma)
    elif utype == 'S-G':
        [gamma, cmin] = params
        u = ((c-cmin)**(1-gamma) - 1.)/(1-gamma)
        mu = (c-cmin)**(-gamma)
    elif utype == 'exponential':
        [a] = params
        u = -np.exp(-a*c)
        mu = np.exp(-a*c) * a
    elif utype == 'HARA':
        [gamma, a, b] = params
        u = ((1-gamma)/gamma) * (a*c/(1-gamma) + b)**gamma
        mu = (a*c/(1-gamma) + b)**(gamma-1) * a
    else:
        u = np.log(c)   
        mu = 1/c    
    
    return u, mu


def generate(distype, utype):

    # set up support for consumption
    eps = .001
    clow = 20000.
    chigh = 30000.
    cnum = 100
    cvec = np.linspace(clow+eps, chigh+eps, num = cnum+1)
    incr = cvec[1] - cvec[0]
    
    # set up PDF
    if distype == 'beta':
        a = 2
        b = 3
        pdf = stats.beta.pdf((cvec-clow)/(chigh-clow), a, b)
    elif distype == 'normal':
        mu = (chigh+clow)/3
        sigma = (chigh-clow)/3
        pdf = stats.norm.pdf((cvec-mu)/sigma)
    else:
        pdf = np.ones(cnum+1)
        for i in range(0, cnum+1):
            if cvec[i] < clow+eps:
                pdf[i] = 0.
            elif cvec[i] > chigh+eps:
                pdf[i] = 0.
            else:
                pdf[i] = 1.
        
    # convert PDF to probabilities
    prob = pdf/np.sum(pdf)    
    
    # find utility
    params = []
    if utype == 'CRRA':
        gamma = 2.
        params = [gamma]
    elif utype == 'S-G':
        gamma = 2.
        cmin = 10000
        params = [gamma, cmin]
    elif utype == 'exponential':
        a = 1/1000.
        params = [a]
    elif utype == 'HARA':
        a = 1.
        b = 1000.
        gamma = .5
        params = [gamma, a, b]
    uvec, muvec = util(cvec, utype, params)
    
    # find expected values
    Ec = np.sum(cvec*prob)
    Eu = np.sum(uvec*prob)
    uEc, muEc = util(Ec, utype, params)
    
    # find useful values
    maxprob = np.max(prob)
    maxu = np.max(uvec)
    minu, minmu = util(clow, utype, params)
    Ec1 = np.array([Ec, Ec])
    prob1 = np.array([0., maxprob])
    Eu1 = np.array([Eu, Eu])
    Eu2 = np.array([uvec[0], uEc])
    Ec3 = np.array([cvec[0], Ec])
    Eu3 = np.array([uEc, uEc])
    Eu4 = np.array([Eu, Eu])
    
    # # plot PDFs
    # fig1 = plt.figure
    # plt.subplot(1,2,1)
    # plt.plot(cvec, prob, 'b+')
    # plt.plot(Ec1, prob1, 'b:')
    # plt.xlabel('Consumption')
    # plt.ylabel('Probability '+distype)
    # plt.subplot(1,2,2)
    # plt.plot(uvec, prob, 'b+')
    # plt.plot(Eu1, prob1, 'b:')
    # plt.xlabel('Utility '+utype)
    # plt.yticks([])
    # plt.xlim([minu, maxu])
    # plt.show()
    
    # # plot utility function
    # fig2 = plt.figure
    # plt.plot(cvec[1:], uvec[1:], 'b')
    # plt.plot(Ec1, Eu2, 'b:')
    # plt.plot(Ec3, Eu3, 'b:')
    # plt.plot(Ec3, Eu4, 'b--')
    # plt.xlabel('Consumption')
    # plt.ylabel('Utility '+utype)
    # plt.show
    
    # parameters for the lottery
    cost = 2
    odds = 300000000
    #ratio = 1.1
    payoff = 1500000000
    Epayoff = .5*payoff/odds
    # print(Epayoff)
    
    # play and lose lottery
    cveclose = np.linspace(clow+eps-cost, chigh+eps-cost, num = cnum+1)
    Eclose = np.sum(cveclose*prob)
    uveclose, muveclose = util(cveclose, utype, params)
    Eulose = np.sum(uveclose*prob)
    Emulose = np.sum(muveclose*prob)
    
    # play and win lottery
    cvecwin = np.linspace(clow+eps-cost+payoff, chigh+eps-cost+payoff, num = cnum+1)
    Ecwin = np.sum(cvecwin*prob)
    uvecwin, muvecwin = util(cvecwin, utype, params)
    Euwin = np.sum(uvecwin*prob)
    Emuwin = np.sum(muvecwin*prob)
    
    # expected values
    Eclottery = ((odds-1.)/odds) * Eclose + (1./odds) * Ecwin
    cdiff = Eclottery - Ec
    Eulottery = ((odds-1.)/odds) * Eulose + (1./odds) * Euwin
    udiff = Eulottery - Eu
    
    # marginal calculations
    Margc = ((odds-1.)/odds) * (-2) + (1./odds) * (-2+payoff)
    Margu = .5 * ( -((odds-1.)/odds) * 2 *Emulose + (1./odds) * payoff * Emuwin)
    
    # print(cdiff, udiff, 2*Margu)
    # print(cdiff/Ec, udiff/Eu, 2*Margu/Eu)
    
    results = np.array([[utype, distype, Margc, 2*Margu, cdiff, udiff, cdiff/Ec, udiff/Eu]])
    
    return results



# Main program
dislist = ['beta', 'normal', 'uniform']
ulist = ['CRRA', 'S-G', 'exponential', 'HARA', 'log']

resultarray = np.empty([len(dislist)*len(ulist), 8], dtype=object)


i=0
for utype in ulist:
    for distype in dislist:
        results = generate(distype, utype)
        resultarray[i,:] = results
        i = i + 1