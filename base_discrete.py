from pyswarms.base.base_discrete import DiscreteSwarmBase
from pyswarms.discrete import binary
import numpy as np
from scipy.spatial import cKDTree
from random import randint


class PerezPSO(DiscreteSwarmBase):

    def assertions(self):
        """Assertion method to check various inputs.

        Raises
        ------
        KeyError
            When one of the required dictionary keys is missing.
        ValueError
            When the number of neighbors is not within the range
                :code:`[0, n_particles]`.
            When the p-value is not in the list of values :code:`[1,2]`.
        """
        super(PerezPSO, self).assertions()

        if not all(key in self.options for key in ('k', 'p')):
            raise KeyError('Missing either k or p in options')
        if not 0 <= self.k <= self.n_particles:
            raise ValueError('No. of neighbors must be between 0 and no. of'
                             'particles.')
        if self.p not in [1, 2]:
            raise ValueError('p-value should either be 1 (for L1/Minkowski)'
                             'or 2 (for L2/Euclidean).')

    def __init__(self, n_particles, dimensions, alpha, options, velocity_clamp=None):
        """Initializes the swarm.

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        velocity_clamp : tuple (default is :code:`None`)
            a tuple of size 2 where the first entry is the minimum velocity
            and the second entry is the maximum velocity. It
            sets the limits for velocity clamping.
        options : dict with keys :code:`{'c1', 'c2', 'k', 'p'}`
            a dictionary containing the parameters for the specific
            optimization technique
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
                * k : int
                    number of neighbors to be considered. Must be a
                    positive integer less than :code:`n_particles`
                * p: int {1,2}
                    the Minkowski p-norm to use. 1 is the
                    sum-of-absolute values (or L1 distance) while 2 is
                    the Euclidean (or L2) distance.
        """
        # Initialize logger
        # self.logger = logging.getLogger(__name__)

        binary = False
        # Assign k-neighbors and p-value as attributes
        self.k, self.p = options['k'], options['p']
        # Initialize parent class
        super(PerezPSO, self).__init__(n_particles, dimensions, binary,
                                        options, velocity_clamp)
        # Invoke assertions
        self.assertions()
        # Initialize the resettable attributes
        self.reset()
        # Set initial glo
        self.glo = np.full((1, self.dimensions), np.inf)
        self.glo_cost = np.inf
        self.y = np.full(self.n_particles, 0)
        self.alpha = alpha
        self.loc_pos = 0

    def optimize(self, objective_func, iters, print_step=1, verbose=1):
        """Optimizes the swarm for a number of iterations.

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : function
            objective function to be evaluated
        iters : int
            number of iterations
        print_step : int (the default is 1)
            amount of steps for printing into console.
        verbose : int  (the default is 1)
            verbosity setting.

        Returns
        -------
        tuple
            the local best cost and the local best position among the
            swarm.
        """
        for i in range(iters):
            # Compute cost for current position and personal best
            current_cost = objective_func(self.pos)

            # Obtain the indices of the best position for each
            # neighbour-space, and get the local best cost and
            # local best positions from it.
            nmin_idx = self._get_neighbors(current_cost)  # get index of loc for each neighborhood of the cur position
            self.best_cost = current_cost[nmin_idx]  # the loc optimum cost for each particle
            if np.abs(current_cost).min() < self.glo_cost:
                pos_min_index = np.where(current_cost == current_cost.min())[0][0]  # index of pos min
                self.glo = self.pos[pos_min_index]
                self.glo_cost = current_cost.min()

            # Get the local min realative to each point
            self.loc_pos = self.pos[nmin_idx]
            self.y = self._get_y(self.loc_pos)

            # Perform position velocity update
            self._update_velocity()  # must be called first
            self._update_position()

            care = r"""
Cur_cost: {}
loc_pos: {}
nmin_idx: {}
y: {}
velocity: {}
position: {}
            """.format(current_cost, self.loc_pos, nmin_idx, self.y, self.velocity, self.pos)

            if i % print_step == 0:
                print(self.y + self.velocity)
                print(care + "\n\n")
                if all_eq(self.pos):
                    break

            if self.glo_cost == 0:
                print(i)
                break

        # Obtain the final best_cost and the final best_position
        # final_best_cost_arg = np.argmin(self.best_cost)
        # final_best_cost = np.min(self.best_cost)
        # final_best_pos = self.best_pos[final_best_cost_arg]
        return self.glo_cost, self.glo

    def _get_neighbors(self, pbest_cost):
        """Helper function to obtain the best position found in the
        neighborhood. This uses the cKDTree method from :code:`scipy`
        to obtain the nearest neighbours

        Parameters
        ----------
        pbest_cost : numpy.ndarray of size (n_particles, )
            the cost incurred at the historically best position. Will be used
            for mapping the obtained indices to its actual cost.

        Returns
        -------
        array of size (n_particles, ) dtype=int64
            indices containing the best particles for each particle's
            neighbour-space that have the lowest cost
        """
        # Use cKDTree to get the indices of the nearest neighbors
        tree = cKDTree(self.pos)
        _, idx = tree.query(self.pos, p=self.p, k=self.k)

        # Map the computed costs to the neighbour indices and take the
        # argmin. If k-neighbors is equal to 1, then the swarm acts
        # independently of each other.
        if self.k == 1:
            # The minimum index is itself, no mapping needed.
            best_neighbor = pbest_cost[idx][:, np.newaxis].argmin(axis=1)
        else:
            idx_min = pbest_cost[idx].argmin(axis=1)
            best_neighbor = idx[np.arange(len(idx)), idx_min]
        return best_neighbor

    def _update_velocity(self):
        """Updates the velocity matrix of the swarm.

        This method updates the attribute :code:`self.velocity` of
        the instantiated object. It is called by the
        :code:`self.optimize()` method.
        """
        # Define the hyperparameters from options dictionary
        c1, c2, w = self.options['c1'], self.options['c2'], self.options['w']

        # Compute for cognitive and social terms
        cognitive = (c1 * np.random.uniform(0, 1) * (-1 - self.y))
        social = (c2 * np.random.uniform(0, 1)
                  * (1 - self.y))
        temp_velocity = (w * self.velocity) + cognitive + social

        # Create a mask to clamp the velocities
        if self.velocity_clamp is not None:
            # Create a mask depending on the set boundaries
            min_velocity, max_velocity = self.velocity_clamp[0], \
                                         self.velocity_clamp[1]
            _b = np.logical_and(temp_velocity >= min_velocity,
                                temp_velocity <= max_velocity)
            # Use the mask to finally clamp the velocities
            self.velocity = np.where(~_b, self.velocity, temp_velocity)
        else:
            self.velocity = temp_velocity

    def _update_position(self):
        """Updates the position matrix of the swarm.

        This method updates the attribute :code:`self.pos` of
        the instantiated object. It is called by the
        :code:`self.optimize()` method.
        """
        del self.pos
        next_pos = np.random.randint(-1000, 1000, size=self.swarm_size)
        _decision = self.y + self.velocity
        # print("des: {}".format(_decision))
        # mext_pos = np.where(_decision > self.alpha, self.glo, next_pos)
        # next_pos = np.where(_decision < self.alpha, self.loc_pos, next_pos)
        for i in range(self.n_particles):
            if _decision[i] > self.alpha:
                next_pos[i] = self.glo
            elif _decision[i] < -self.alpha:
                next_pos[i] = self.loc_pos[i]
        self.pos = next_pos

    def _get_y(self, loc):
        _y = np.array([])
        for i in range(self.n_particles):
            if np.array_equal(self.glo, self.pos[i]):
                _y = np.concatenate((_y, [1]))
            elif np.array_equal(loc[i], self.pos[i]):
                _y = np.concatenate((_y, [-1]))
            else:
                _y = np.concatenate((_y, [0]))
        return _y

    def _sigmoid(self, x):
        """Helper sigmoid function.

        Inputs
        ------
        x : numpy.ndarray
            Input vector to compute the sigmoid from

        Returns
        -------
        numpy.ndarray
            Output sigmoid computation
        """
        return 1 / (1 + np.exp(x))


def all_eq(position):
    first = position[0]
    for x in position:
        if not np.array_equal(x, first):
            return False
    return True


if __name__ == "__main__":
    from Particle import majic_func as obj_func
    from pyswarms.utils.environments import PlotEnvironment
    test = PerezPSO(10000, 16, 0.3, {"k": 50, 'c1': 0.8, 'c2': 0.2, 'w': 0.75, 'p': 2})
    test.pos = np.random.randint(-1000, 1000, size=test.swarm_size)
    test.velocity = np.full(test.n_particles, 0)

    #plt_env = PlotEnvironment(test, test_func2, 1000)
    #plt_env.plot_cost(figsize=(-10, 10))
    #plt_env.plot_particles2D(limits=((-10, 10), (-10, 10)))

    # print(test.pos)
    print(test.optimize(obj_func, iters=20000, print_step=10))
