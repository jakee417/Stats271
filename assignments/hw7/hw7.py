import numpy as np
import numpy.random as npr

event_data = np.load('./assignments/hw7/event_data.npy', allow_pickle = True)
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
        T, K = log_likelihoods.shape
        # alpha.shape => (K,)
        alphas = [self.initial_dist]
        likelihoods = np.exp(log_likelihoods)
        marginal_ll = 0
        for t in range(T-1):
            # normalize for numerical stability
            A = alphas[t].dot(likelihoods[t, :])
            marginal_ll += np.log(A)
            # (K, K) dot ((K,) * (K,))
            alphas[t+1] = (1 / A) * self.transition_matrix.T.dot(alphas[t] * likelihoods[t, :])
        # FixMe: Why can we interpret alphas as p(zt=k|x1:t-1) and At = p(xt|x1:t-1)?
        alphas = np.concatenate(alphas)
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
        T, K = log_likelihoods.shape
        # beta.shape => (K,)
        betas = [np.ones(K)]
        likelihoods = np.exp(log_likelihoods)
        for t in range(T-2, -1, -1):
            # FixMe: Insert normalizing constant for alphas
            # (K, K) dot ((K,) * (K,))
            betas[t] = self.transition_matrix.dot(betas[t+1] * likelihoods[t+1, :])
        res = np.concatenate(betas)
        assert res.shape == (T, K)
        return res


    def e_step(self, parameters):
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
            # compute log likelihoods for each time step
            # from the initialized parameters
            log_likelihoods = None
            alphas, temp_marginal_ll = self.forward_pass(log_likelihoods)
            # Sum conditionally independent "events"
            marginal_ll += temp_marginal_ll
            betas = self.backward_pass(log_likelihoods)
            # compute expectations
            likelihoods = np.exp(log_likelihoods)
            temp = alphas * likelihoods * betas
            expectations.append(temp / (np.sum(temp, axis=1)))
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
        pass


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
        # combine all data and apply k-means clustering for
        # initial parameters
        self.parameters = None

        self.events_cache = {
            1: self.parameters
        }

        i = 0
        while i < 10:
            # FixMe: What are good convergence checks?
            expectations, marginal_ll = self.e_step(parameters)
            parameters = self.m_step(expectations)
            i += 1
        return parameters


# FixMe: How do we initialize the parameters?
def random_args(num_timesteps, num_states,
                offset=0, scale=1):
    rng = npr.RandomState(0)
    log_likes = offset + scale * rng.randn(num_timesteps, num_states)
    return log_likes
