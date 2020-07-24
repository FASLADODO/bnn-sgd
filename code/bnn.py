import os
import numpy as np
import matplotlib.pyplot as plt

from numpy.random import randn, rand
from sklearn.linear_model import LogisticRegression


def logistic_sigmoid(x):
    return 1. / (1. + np.exp(-x))


def log_logistic_sigmoid(x):
    out = np.empty_like(x)
    out[x>0.] = -np.log(1. + np.exp(-x[x>0.]))
    out[x<=0.] = x[x<=0.] - np.log(1. + np.exp(x[x<=0.]))
    return out


def bnn_logreg_full(Y, X, mu=None, C=None, niter=100, kappa=1., tau=0.,
                     m0=None, S0=None):

    N, p = X.shape
    if mu is None:
        mu = np.zeros(p)
    if C is None:
        C = np.eye(p)
    if m0 is None:
        m0 = np.zeros(p)
    if S0 is None:
        S0 = np.eye(p)

    L0 = np.linalg.inv(S0)
    L0m0 = np.dot(L0, m0)
    iesbo = np.empty(niter)
    rho = (1. + tau) ** (-kappa)

    for it in xrange(niter):
        
        z = randn(p)
        th = np.dot(C, z) + mu

        w = -Y * np.dot(X, th)
        YX = Y[:,np.newaxis]*X
        S = logistic_sigmoid(w)
        grad_logreg = np.sum(S[:,np.newaxis] * YX, axis=0) \
                       - np.dot(L0, th) + L0m0

        mu += rho * grad_logreg
        dC = np.outer(grad_logreg, z)
        dC = np.tril(dC) + np.diag(1./np.diag(C))
        C += 0.1*rho * dC
        C = np.tril(C)
        keep = np.diag(C).copy()
        keep[keep<=1e-4] = 1e-4
        C = C + (np.diag(keep - np.diag(C)))

        rho = (1. + it + tau) ** (-kappa)

        iesbo[it] = np.sum(log_logistic_sigmoid(-w), axis=0) \
                    - 0.5*np.dot(th, np.dot(L0, th)) \
                    + np.dot(m0, np.dot(L0, th))
        iesbo[it] += np.sum(np.log(np.diag(C)))
        iesbo[it] += 0.5*p + 0.5*p*np.log(2.*np.pi)

    return mu, C, iELB


if __name__ == "__main__":

    np.random.seed(42)
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    dpath = "~/work/data/"
    df = pd.read_csv(os.path.expanduser(dpath))
    X_full = df.ix[:,df.columns != 'CLASS'].values
    N, p = X_full.shape
    tmp = np.empty((N, p+1))
    tmp[:,0] = 1.
    tmp[:,1:] = X_full
    X_full = tmp
    Y_full = df['CLASS'].values
    Y_full = 2*Y_full - 1

    X_full -= np.mean(X_full, axis=0)
    X_full[:,1:] /= np.std(X_full[:,1:], axis=0)

    X, X_test, Y, Y_test = train_test_split(X_full, Y_full, test_size=0.2,
                                            random_state=8675309)
    N, p = X_full.shape
    N_test = X_test.shape[0]

    niter = 5000
    wlen = 200
    mu, C, F = bnn_logreg_full(Y, X, mu=None, C=np.eye(p), niter=niter,
                                kappa=.7, tau=50)

    Fsm = np.zeros(niter-wlen)
    for i in xrange(Fsm.shape[0]):
        Fsm[i] = np.mean(F[i:i+wlen])

    lr = LogisticRegression(penalty='l2', fit_intercept=False, C=0.5)
    lr.fit(X, Y)

    plt.plot(Fsm)
    plt.title('Smoothed instantaneous esbo')

    dr_tr = (np.round(logistic_sigmoid(np.dot(X, mu))) - 0.5)*2.
    opr_tr = lr.predict(X)
    dr_te = (np.round(logistic_sigmoid(np.dot(X_test, mu))) - 0.5)*2.
    opr_te = lr.predict(X_test)

    print "Train Accuracy:"
    print "  bnn: %.2f" % (1. - np.sum(np.abs(Y - dr_tr) > 0.) / float(N),)
    print "  LR: %.2f" % (1. - np.sum(np.abs(Y - opr_tr) > 0.) / float(N),)

    print "Test Accuracy:"
    print "  bnn: %.2f" % (1. - np.sum(np.abs(Y_test - dr_te) > 0.) / float(N_test),)
    print "  LR: %.2f" % (1. - np.sum(np.abs(Y_test - opr_te) > 0.) / float(N_test),)
