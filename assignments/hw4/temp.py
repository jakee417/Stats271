import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
from collections import Counter

def gibbs(X,
          K=2, 
          n_iter=100, 
          alpha=np.ones(2), 
          m0=np.zeros(3), 
          kappa0=1.0, 
          nu0=3.0, 
          Psi0=np.eye(3)):
    """
    Run the Gibbs sampler for a mixture of Gaussians with a NIW prior.

    Input:
    - X: Matrix of size (N, D). Each row of X stores one data point
    - K: the desired number of clusters in the model. Default: 2
    - n_iter: number of iterations of Gibbs sampling. Default: 100
    - alpha: hyperparameter of the Dirichlet prior on \pi.
    - m0, kappa0, nu0, Psi0, hyperparameters of normal-inverse-Wishart prior.

    Returns: 
    - log joint probability for each iteration
    - samples of parameters and assignments over iterations

    You will use these values later on for convergence diagnostics.
    """
    N, D = X.shape
    clusters = range(K)
    
    # init random assignments
    z = np.random.randint(low=0, high=K, size=N)

    # init params
    kappa = dict.fromkeys(clusters, kappa0)
    nu = dict.fromkeys(clusters, nu0)
    Psi = dict.fromkeys(clusters, Psi0)
    m = dict.fromkeys(clusters, m0)
    alpha = np.ones(K)
    
    # init caches
    kappa_cache = []
    nu_cache = []
    Psi_cache = []
    m_cache = []
    alpha_cache = []
    z_cache = []
    n_k_cache = []
    Sigma_cache = []
    pi_cache = []
    mu_cache = []
    ll_cache = []
    
    # Define helper function for vectorized ops
    def random_indexes(prob_matrix, items):
        s = prob_matrix.cumsum(axis=0)
        r = np.random.rand(prob_matrix.shape[1])
        k = (s < r).sum(axis=0)
        return items[k]
    
    for iteration in range(n_iter):
        # Get counts of z's
        counts = dict(Counter(z))
        N_k = dict.fromkeys(clusters, 0)
        N_k.update(counts)
        n_k = np.array([i[1] for i in sorted(N_k.items(), key = lambda kv:kv[0])])
        #print(f'label counts for iteration {iteration}: {n_k}')
        
        if iteration == n_iter-1:
            print('last round')
        
        ll = 0
        Sigma = {}
        mu = {}
        
        # sample pi
        pi = ss.dirichlet.rvs(alpha + n_k)
        ll += ss.dirichlet.logpdf(pi, alpha + n_k)

        
        # update mu_k, Psi_k
        for item in N_k.items():
            k, n = item
            
            if n == 0:
                Sigma[k] = ss.invwishart.rvs(df=nu[k], scale=Psi[k])
                mu[k] = ss.multivariate_normal.rvs(mean=m[k], cov=Sigma[k] / kappa[k])
                continue
            
            # select observations
            ind = z == k
            x = X[ind,]

            # update easy ones
            kappa[k] = kappa0 + n
            nu[k] = nu0 + n
            
            # helper computations
            xbar = np.mean(x, axis=0)
            S = (x - xbar).T.dot(x - xbar) # scatter matrix
            
            # mean scatter matrix            
            Sbar = ((kappa0 * n)/(kappa0 + n)) * \ss.multivariate_normal.logpdf(x, mu[k], Sigma[k]).sum()
                (xbar - m0)[np.newaxis,...].T \
                    .dot((xbar - m0)[np.newaxis,...])
                    
            # update eta params
            Psi[k] = Psi0 + S + Sbar
            m[k] = (kappa0 * m0 + n * xbar) / (kappa0 + n)
            
            Sigma[k] = ss.invwishart.rvs(df=nu[k], scale=Psi[k])
            mu[k] = ss.multivariate_normal.rvs(mean=m[k],
                                               cov=Sigma[k] / kappa[k])
            
            ll += ss.invwishart.logpdf(Sigma[k], df=nu0, scale=Psi0)
            ll += ss.multivariate_normal.logpdf(mu[k], mean=m0,
                                                cov=Sigma[k] / kappa0)
            
            ll += ss.multivariate_normal.logpdf(x, mu[k], Sigma[k]).sum()
            

        # sample z
        p_x = np.array([ss.multivariate_normal.pdf(X, mu[i], Sigma[i]) for i in clusters])
        p_z = (pi.T * p_x) / np.sum(pi.T * p_x, axis=0)
        z = random_indexes(p_z, np.array(clusters))        
        ll += ss.bernoulli.logpmf(z, pi[0, 1]).sum()
                            
        # collect artifacts
        mu_cache.append(mu)
        Sigma_cache.append(Sigma)
        pi_cache.append(pi)
        n_k_cache.append(n_k)
        z_cache.append(z)
        kappa_cache.append(kappa)
        nu_cache.append(nu)
        Psi_cache.append(Psi)
        m_cache.append(m)
        alpha_cache.append(alpha)
        ll_cache.append(ll)
        
    return {
        'z':z_cache,
        'kappa':kappa_cache,
        'nu':nu_cache,
        'Psi':Psi_cache,
        'm':m_cache,
        'alpha':alpha_cache,
        'n_k':n_k_cache,
        'mu':np.array([list(i.values()) for i in mu_cache]),
        'Sigma':np.array([list(i.values()) for i in Sigma_cache]),
        'pi':pi_cache,
        'll':ll_cache
    }

X = np.vstack([
    ss.multivariate_normal.rvs(mean = [10]* 3, cov = 0.1 * np.eye(3), size = 50),
    ss.multivariate_normal.rvs(mean = [-10]* 3, cov = 0.1 * np.eye(3), size = 50)
    ])

res = gibbs(X, n_iter=101)
print('success')

for i in range(2):
    for j in range(3):
        plt.plot(np.array(res['mu'])[:, i, j], alpha = 0.6)
plt.show()

plt.plot(res['ll'])
plt.show()

print(res['Sigma'][-1])
