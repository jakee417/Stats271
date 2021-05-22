import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt

event_data = np.load('/home/jake/PycharmProjects/Stats271/assignments/hw7/event_data.npy', allow_pickle=True)
print(event_data.shape)
event_data = list(event_data)

print("number of events: ", len(event_data))
print("average event length: ", np.mean([len(xs) for xs in event_data]))
print("total number of time steps", np.sum([len(xs) for xs in event_data]))


class HiddenMarkovModel:
    def __init__(self, num_states, epsilon=0.01):
        self.epsilon = epsilon
        self.num_states = num_states
        self.initial_distribution = np.ones(self.num_states) / self.num_states
        self.transition_matrix = np.diag(np.ones(self.num_states))
        self.transition_matrix = np.where(self.transition_matrix == 0,
                                          (epsilon / (self.num_states - 1)),
                                          1 - self.epsilon)
        self.normals = {}
        self.data = None
        self.events = None
        self.marginals = []

    def forward_pass(self, log_likelihoods):
        """Perform the forward pass and return the forward messages for
        a single "event".

        In the descriptions below, let K denote the number of discrete states
        and T the number of time steps.

        Parameters
        ---
        initial_dist: (K,) array with initial state probabilities
        transition_matrix: (K, K) array where each row is a transition probability
        log_likelihoods: (T, K) array with entries log p(x_t | z_t=k)

        Returns
        ---
        alphas: (T, K) array of forward messages
        marginal_ll: real-valued scalar, log p(x_{1:T})
        """
        # alpha.shape => (K,)
        alphas = [self.initial_distribution]
        likelihoods = np.exp(log_likelihoods)
        marginal_ll = 0
        try:
            T, K = log_likelihoods.shape
        except:
            K = len(log_likelihoods)
            T = 1
            A = alphas[0].dot(likelihoods)
            alpha = (1 / A) * self.transition_matrix.T.dot(alphas[0] * likelihoods)
            return alpha.reshape(T, K), np.log(A)
        for t in range(T - 1):
            # normalize for numerical stability
            A = alphas[t].dot(likelihoods[t, :])
            marginal_ll += np.log(A)
            # (K, K) dot ((K,) * (K,))
            alphas.append((1 / A) * self.transition_matrix.T.dot(alphas[t] * likelihoods[t, :]))
        alphas = np.vstack(alphas)
        assert alphas.shape == (T, K)
        return alphas, marginal_ll

    def backward_pass(self, log_likelihoods):
        """Perform the backward pass and return the backward messages for
        a single "event".

        Parameters
        ---
        transition_matrix: (K, K) array where each row is a transition probability
        log_likelihoods: (T, K) array with entries log p(x_t | z_t=k)

        Returns
        ---
        betas: (T, K) array of backward messages
        """
        likelihoods = np.exp(log_likelihoods)
        try:
            T, K = log_likelihoods.shape
        except:
            K = len(log_likelihoods)
            T = 1
            B = np.ones(K).dot(likelihoods)
            beta = ((1 / B) * self.transition_matrix.dot(np.ones(K) * likelihoods))
            return beta.reshape(T, K)
        betas = [1] * T
        # beta.shape => (K,)
        betas[T - 1] = np.ones(K)
        for t in range(T - 1, -1, -1):
            B = betas[t].dot(likelihoods[t, :])
            # (K, K) dot ((K,) * (K,))
            betas[t - 1] = ((1 / B) * self.transition_matrix.dot(betas[t] * likelihoods[t, :]))
        betas = np.vstack(betas)
        assert betas.shape == (T, K)
        return betas

    def e_step(self):
        """Run the E step for each event. First compute the log likelihoods
        for each time step and discrete state using the given data and parameters.
        Then run the forward and backward passes and use the output to compute the
        posterior marginals, and use marginal_ll to compute the marginal likelihood.

        Parameters
        ---
        data: list of (T, 20) arrays with player positions over time for each event
        parameters: a data structure containing the model parameters; i.e. the
            initial distribution, transition matrix, and Gaussian means and
            covariances.

        Returns
        ---
        expectations: list of (T, K) arrays of marginal probabilities
            p(z_t = k | x_{1:T}) for each event.
        marginal_ll: marginal log probability p(x_{1:T}). This should go up
            each iteration!
        """
        expectations = []
        marginal_ll = 0
        # Run an E-step for each event
        for event in range(self.events):
            temp_data = self.data[event]
            T, K = temp_data.shape
            # compute log likelihoods for each time step
            # from the initialized parameters
            # log_likelihoods: (T, K) array with entries log p(x_t | z_t=k)
            log_likelihoods = []
            for normal in self.normals:
                log_likelihoods.append(self.normals[normal].logpdf(temp_data)[..., None])
            log_likelihoods = np.hstack(log_likelihoods)
            alphas, temp_marginal_ll = self.forward_pass(log_likelihoods)
            # Sum conditionally independent "events"
            marginal_ll += temp_marginal_ll
            betas = self.backward_pass(log_likelihoods)
            # compute expectations
            likelihoods = np.exp(log_likelihoods)
            temp = alphas * likelihoods * betas
            expectation = temp / (np.sum(temp, axis=1)[..., None])
            assert expectation.shape == (T, K)
            expectations.append(expectation)
        assert len(expectations) == len(self.data)
        return expectations, marginal_ll

    def m_step(self, expectations):
        """Solve for the Gaussian parameters that maximize the expected log
        likelihood.

        Note: you can assume fixed initial distribution and transition matrix as
        described in the markdown above.

        Parameters
        ----------
        data: list of (T, 20) arrays with player positions over time for each event
        expectations: list of (T, K) arrays with marginal state probabilities from
            the E step.

        Returns
        -------
        parameters: a data structure containing the model parameters; i.e. the
            initial distribution, transition matrix, and Gaussian means and
            covariances.
        """
        # Consolidate all timesteps into one set of parameters
        # collapse over all E-step event outputs
        # global set of parameters that we are maintaining
        self.normals = {}
        total_data = np.concatenate(self.data)
        total_expectations = np.concatenate(expectations)
        for i in range(self.num_states):
            weights = total_expectations[:, i, None]
            psi_k_2 = (weights * total_data).sum(axis=0)
            psi_k_1 = (weights * total_data).T @ total_data
            psi_k_3 = np.sum(weights)
            bk = psi_k_2 / psi_k_3
            Qk = (1 / psi_k_3) * (
                psi_k_1
                - psi_k_2[..., None] @ psi_k_2[..., None].T
                / psi_k_3
            )
            self.normals[i] = multivariate_normal(mean=bk, cov=Qk)

    def init_params(self):
        total_data = np.concatenate(self.data)
        labels = np.array(list(range(0, self.num_states)) * int(np.ceil(len(total_data) / self.num_states)))[:len(total_data)]
        for i in range(self.num_states):
            temp = total_data[labels == i]
            bk = np.mean(temp, axis=0)
            Qk = (1 / len(temp)) * (
                    (temp.T @ temp)
                    - (temp.sum(axis=0)[..., None] @ temp.sum(axis=0)[..., None].T)
                    / len(temp)
            )
            self.normals[i] = multivariate_normal(mean=bk, cov=Qk)

    def marginal_likelihood(self, data):
        """Compute marginal log-likelihood on dataset"""
        if isinstance(data, list):
            events = len(data)
        else:
            events = 1
        marginal_ll = 0
        # Run an E-step for each event
        for event in range(events):
            if events != 1:
                temp_data = data[event]
            else:
                temp_data = data
            T, K = temp_data.shape
            # compute log likelihoods for each time step
            # from the initialized parameters
            # log_likelihoods: (T, K) array with entries log p(x_t | z_t=k)
            log_likelihoods = []
            for normal in self.normals:
                log_likelihoods.append(self.normals[normal].logpdf(temp_data)[..., None])
            log_likelihoods = np.hstack(log_likelihoods)
            alphas, temp_marginal_ll = self.forward_pass(log_likelihoods)
            # Sum conditionally independent "events"
            marginal_ll += temp_marginal_ll
        return marginal_ll

    def fit_hmm(self, data):
        """Fit an HMM using the EM algorithm above. You'll have to initialize the
        parameters somehow; k-means often works well. You'll also need to monitor
        the marginal likelihood and check for convergence.

        Returns
        -------
        lls: the marginal log likelihood over EM iterations
        parameters: the final parameters
        """
        self.data = data
        self.events = len(self.data)
        # combine all data and apply k-means clustering for inital params
        self.init_params()
        self.marginals = []
        i = 0
        threshold = np.inf
        while threshold > 1e-1:
            expectations, marginal_ll = self.e_step()
            self.marginals.append(marginal_ll)
            if len(self.marginals) > 1:
                threshold = self.marginals[-1] - self.marginals[-2]
            self.m_step(expectations)
            print(f"{i}:{marginal_ll}")
            i += 1
        return self.normals


HMM = HiddenMarkovModel(num_states=20)
HMM.fit_hmm(event_data)
plt.plot(HMM.marginals)
plt.show()
HMM.marginal_likelihood(event_data[1:3])


