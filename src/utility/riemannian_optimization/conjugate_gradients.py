from .. import backend

"""
This file implements the Conjugate Gradients algorithm on Riemannian manifolds.
Sources:
[1] P.-A. Absil, Robert Mahony, Rodolphe Sepulchre: "Optimization Algorithms on Matrix Manifolds", https://press.princeton.edu/absil
[2] Markus Hauru, Maarten Van Damme, Jutho Haegeman: "Riemannian optimization of isometric tensor networks", https://scipost.org/10.21468/SciPostPhys.10.2.040
[3] James Townsend, Niklas Koep, Sebastian Weichwald: "Pymanopt: A Python Toolbox for Optimization on Manifolds using Automatic Differentiation", https://arxiv.org/abs/1603.03236
[4] Jonathan Richard Shewchuk, "An Introduction to the Conjugate Gradient Method Without the Agonizing Pain", https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
[5] William W. Hager and Hongchao Zhang, "A SURVEY OF NONLINEAR CONJUGATE GRADIENT METHODS", https://www.cmor-faculty.rice.edu/~yzhang/caam554/pdf/cgsurvey.pdf
[6] Caixia Kou, Wen-Hui Zhang, Wen-Bao Ai, Ya-Feng Liu: "On the use of Powell's restart strategy to conjugate gradient methods"
[7] William W. Hager and Hongchao Zhang, "Algorithm 851: CG DESCENT, a Conjugate Gradient Method with Guaranteed Descent", https://www.math.lsu.edu/~hozhang/papers/cg_compare.pdf
"""

def _beta_hestenes_stiefel(manifold, x_k, x_kp1, grad_k, grad_kp1, mu_k):
    """
    Computes the beta factor by hestenes and stiefel, used for computing the next search direction.

    Parameters
    ----------
    manifold : Manifold class implementing the functions inner_product() and transport()
        instance of the class representing the manifold
    x_k : backend.array_type
        current iterate, element of the manifold
    x_kp1 : backend.array_type
        next iterate, element of the manifold
    grad_k : backend.array_type
        gradient at the current iterate, element of the tangent space of x_k
    grad_kp1 : backend.array_type
        gradient at the next iterate, element of the tangent space of x_kp1
    mu_k : backend.array_type
        current search direction

    Returns
    -------
    beta:
        the computed beta factor
    """
    grad_k_transported = manifold.transport(x_kp1, grad_k)
    y = grad_kp1 - grad_k_transported
    return manifold.inner_product(grad_kp1, y) / manifold.inner_product(mu_k, y)

def _beta_fletcher_reeves(manifold, x_k, x_kp1, grad_k, grad_kp1, mu_k):
    """
    Computes the beta factor by fletcher and reeves, used for computing the next search direction.

    Parameters
    ----------
    manifold : Manifold class implementing the functions inner_product() and transport()
        instance of the class representing the manifold
    x_k : backend.array_type
        current iterate, element of the manifold
    x_kp1 : backend.array_type
        next iterate, element of the manifold
    grad_k : backend.array_type
        gradient at the current iterate, element of the tangent space of x_k
    grad_kp1 : backend.array_type
        gradient at the next iterate, element of the tangent space of x_kp1
    mu_k : backend.array_type
        current search direction

    Returns
    -------
    beta:
        the computed beta factor
    """
    return manifold.inner_product(grad_kp1, grad_kp1) / manifold.inner_product(grad_k, grad_k)

def _beta_polark_riberie(manifold, x_k, x_kp1, grad_k, grad_kp1, mu_k):
    """
    Computes the beta factor by polar and riberie, used for computing the next search direction.

    Parameters
    ----------
    manifold : Manifold class implementing the functions inner_product() and transport()
        instance of the class representing the manifold
    x_k : backend.array_type
        current iterate, element of the manifold
    x_kp1 : backend.array_type
        next iterate, element of the manifold
    grad_k : backend.array_type
        gradient at the current iterate, element of the tangent space of x_k
    grad_kp1 : backend.array_type
        gradient at the next iterate, element of the tangent space of x_kp1
    mu_k : backend.array_type
        current search direction

    Returns
    -------
    beta:
        the computed beta factor
    """
    grad_k_transported = manifold.transport(x_kp1, grad_k)
    y = grad_kp1 - grad_k_transported
    return manifold.inner_product(grad_kp1, y) / manifold.inner_product(grad_k, grad_k)

def _beta_conjugate_descent(manifold, x_k, x_kp1, grad_k, grad_kp1, mu_k):
    """
    Computes the beta factor from the conjugate descent algorithm, used for computing the next search direction.

    Parameters
    ----------
    manifold : Manifold class implementing the functions inner_product() and transport()
        instance of the class representing the manifold
    x_k : backend.array_type
        current iterate, element of the manifold
    x_kp1 : backend.array_type
        next iterate, element of the manifold
    grad_k : backend.array_type
        gradient at the current iterate, element of the tangent space of x_k
    grad_kp1 : backend.array_type
        gradient at the next iterate, element of the tangent space of x_kp1
    mu_k : backend.array_type
        current search direction

    Returns
    -------
    beta:
        the computed beta factor
    """
    return manifold.inner_product(grad_kp1, grad_kp1) / -manifold.inner_product(mu_k, grad_k)

def _beta_liu_storey(manifold, x_k, x_kp1, grad_k, grad_kp1, mu_k):
    """
    Computes the beta factor by liu and storey, used for computing the next search direction.

    Parameters
    ----------
    manifold : Manifold class implementing the functions inner_product() and transport()
        instance of the class representing the manifold
    x_k : backend.array_type
        current iterate, element of the manifold
    x_kp1 : backend.array_type
        next iterate, element of the manifold
    grad_k : backend.array_type
        gradient at the current iterate, element of the tangent space of x_k
    grad_kp1 : backend.array_type
        gradient at the next iterate, element of the tangent space of x_kp1
    mu_k : backend.array_type
        current search direction

    Returns
    -------
    beta:
        the computed beta factor
    """
    grad_k_transported = manifold.transport(x_kp1, grad_k)
    y = grad_kp1 - grad_k_transported
    return manifold.inner_product(grad_kp1, y) / -manifold.inner_product(mu_k, grad_k)

def _beta_dai_yuan(manifold, x_k, x_kp1, grad_k, grad_kp1, mu_k):
    """
    Computes the beta factor by dai and yuan, used for computing the next search direction.

    Parameters
    ----------
    manifold : Manifold class implementing the functions inner_product() and transport()
        instance of the class representing the manifold
    x_k : backend.array_type
        current iterate, element of the manifold
    x_kp1 : backend.array_type
        next iterate, element of the manifold
    grad_k : backend.array_type
        gradient at the current iterate, element of the tangent space of x_k
    grad_kp1 : backend.array_type
        gradient at the next iterate, element of the tangent space of x_kp1
    mu_k : backend.array_type
        current search direction

    Returns
    -------
    beta:
        the computed beta factor
    """
    grad_k_transported = manifold.transport(x_kp1, grad_k)
    y = grad_kp1 - grad_k_transported
    mu_k_transported = manifold.transport(x_kp1, mu_k)
    return manifold.inner_product(grad_kp1, grad_kp1) / manifold.inner_product(mu_k_transported, y)

def _beta_hager_zhang(manifold, x_k, x_kp1, grad_k, grad_kp1, mu_k):
    """
    Computes the beta factor by hager and zhang, used for computing the next search direction.

    Parameters
    ----------
    manifold : Manifold class implementing the functions inner_product() and transport()
        instance of the class representing the manifold
    x_k : backend.array_type
        current iterate, element of the manifold
    x_kp1 : backend.array_type
        next iterate, element of the manifold
    grad_k : backend.array_type
        gradient at the current iterate, element of the tangent space of x_k
    grad_kp1 : backend.array_type
        gradient at the next iterate, element of the tangent space of x_kp1
    mu_k : backend.array_type
        current search direction

    Returns
    -------
    beta:
        the computed beta factor
    """
    grad_k_transported = manifold.transport(x_kp1, grad_k)
    y = grad_kp1 - grad_k_transported
    mu_k_transported = manifold.transport(x_kp1, mu_k)
    temp = manifold.inner_product(mu_k_transported, y)
    result = y - 2*mu_k_transported * manifold.inner_product(y, y) / temp
    return manifold.inner_product(result, grad_kp1) / temp

class ConjugateGradientsOptimizer:
    """
    Class implementing the Conjugate Gradients algorithm on Riemannian manifolds. To use this class, first initialize an instance of
    the class with the desired parameters, and then call optimize().
    The iterates are instances of a iterate class, that is responsible for evaluating the cost function and computing gradients.
    For an example see the RenyiAlphaIterate class from src/utility/disentangle/disentangle_renyi_alpha_cg.py.
    """

    def __init__(self, manifold, construct_iterate, beta_rule="hestenes_stiefel", restart_factor=0.9, N_iters=1000, 
            step_size_eps=1e-9, grad_norm_eps=1e-9, ls_contraction_factor=0.5, ls_sufficient_decrease=1e-4, 
            ls_max_iterations=25, ls_initial_step_size=1.0):
        """
        Initializes the class.

        Parameters
        ----------
        manifold : Manifold class implementing the functions inner_product(), norm(), transport() and retract().
            instance of the class representing the manifold
        construct_iterate : function
            function that is used to construct instances of an iterate class, see e.g. src/utility/disentangle/disentangle_renyi_alpha.py.
            The function should take as arguments an element from the manifold, and the previous instance of the iterate class (or None),
            and return a new instance of the iterate class. This allows for a lot of flexibility in how cashing is done to save resources.
            The iterate class must implement functions for computing the value of the cost function and the gradient.
        beta_rule : str, one of {"hestenes_stiefel", "fletcher_reeves", "polark_riberie", "conjugate_descent", "liu_storey", "dai_yuan", "hager_zhang"}, optional
            string for selecting the rule used to compute beta in the CG algorithm. The selection can change the convergence behaviour of CG
            drastically. Default: "hestenes_stiefel".
        restart_factor : float, optional
            constant > 0 for Powell's restart strategy [5]. Setting this to backend.inf disables the restart strategy. Default: 0.9
        N_iters : int, optional
            The maximum number of iterations CG is run for. Default: 1000
        step_size_eps : float, optional
            If the absolute value of the step_size is smaller than this threshhold value, the algorithm is terminated. Default: 1e-9.
        grad_norm_eps : float, optional
            If the norm of the gradient is smaller than this threshhold value, the algorithm is terminated. Default: 1e-9.
        ls_contraction_factor : float, optional
            contraction factor used in the armijo-like line search in each CG iteration. Default: 0.5.
        ls_sufficient_decrease : float, optional
            used for deciding what constitutes a sufficient decrease in the armijo-like line search. Default: 1e-4.
        ls_max_iterations : int, optional
            maximum number of line search iterations. Should be chosen s.t. the step_size ls_contraction_factor**ls_max_iterations
            is close enough to zero to guarantee that the negative gradient is a descent direction.
        ls_initial_step_size : float, optional
            initial step size when the line search algorithm is called in the first CG iteration
        """
        self.manifold = manifold
        self.construct_iterate = construct_iterate
        # Parse beta rule
        if beta_rule == "hestenes_stiefel":
            self.compute_beta = _beta_hestenes_stiefel
        elif beta_rule == "fletcher_reeves":
            self.compute_beta = _beta_fletcher_reeves
        elif beta_rule == "polark_riberie":
            self.compute_beta = _beta_polark_riberie
        elif beta_rule == "conjugate_descent":
            self.compute_beta = _beta_conjugate_descent
        elif beta_rule == "liu_storey":
            self.compute_beta = _beta_liu_storey
        elif beta_rule == "dai_yuan":
            self.compute_beta = _beta_dai_yuan
        elif beta_rule == "hager_zhang":
            self.compute_beta = _beta_hager_zhang
        else:
            raise NotImplementedError(f"beta_rule \"{beta_rule}\" is not implemented.")
        self.restart_factor = restart_factor
        self.N_iters = N_iters
        self.step_size_eps = step_size_eps
        self.grad_norm_eps = grad_norm_eps
        self.ls_contraction_factor = ls_contraction_factor
        self.ls_sufficient_decrease = ls_sufficient_decrease
        self.ls_max_iterations = ls_max_iterations
        self.ls_initial_step_size = ls_initial_step_size

    def _line_search_adaptive(self, iterate, search_direction, slope, cost, old_alpha=None):
        """
        Performs an adaptive armijo-like line search to decide how far along the given search direction we should move.

        Parameters
        ----------
        iterate : instance of iterate class
            the current iterate.
        search_direction : backend.array_type
            the current search direction. Element of the tangent space at the current iterate.
        slpoe : float
            the slope of the cost function at the current iterate when moving along the current search direction.
        cost : float
            the value of the cost function at the current iterate 
        old_alpha : float or None, optional:
            the old value of the final alpha during the last call to this function, or None if this
            is the first time calling this function.

        Returns
        -------        
        step_size : float
            the step size found by the line search algorithm
        new_iterate : instance of iterate class
            the next iterate.
        alpha : float
            the alpha value that can be used fo the old_alpha parameter when calling this function the next time.
        """
        search_direction_norm = self.manifold.norm(search_direction)
        if old_alpha is not None:
            alpha = old_alpha
        else:
            alpha = self.ls_initial_step_size / search_direction_norm

        # Make the step and compute the cost
        new_iterate = self.construct_iterate(self.manifold.retract(iterate.get_iterate(), alpha*search_direction), iterate)
        new_cost = new_iterate.evaluate_cost_function()
        cost_evaluations = 1

        while new_cost > cost - self.ls_sufficient_decrease * alpha * slope and cost_evaluations < self.ls_max_iterations:
                # Reduce step size
                alpha *= self.ls_contraction_factor

                # Make the step and compute the cost
                new_iterate = self.construct_iterate(self.manifold.retract(iterate.get_iterate(), alpha*search_direction), new_iterate)
                new_cost = new_iterate.evaluate_cost_function()
                cost_evaluations += 1

        if new_cost >= cost:
            # If after the maximum number of iterations we still have not managed
            # to decrease the cost, do not do anything
            alpha = 0
            new_iterate = iterate
            new_cost = cost

        step_size = alpha * search_direction_norm

        if cost_evaluations == 2:
            # We expect on average two evaluations. This means it is going well and
            # we can keep the pace
            return step_size, new_iterate, new_cost, alpha
        else:
            # Either things went very well, or we had to backtrack a lot.
            # This probably means that the stepsize is quite small and we can speed up
            return step_size, new_iterate, new_cost, 2*alpha

    def optimize(self, initial_iterate, log_debug_info=False, log_iterates=False):
        """
        Optimizes the given initial iterate with the Conjugate Gradients algorithm.

        Parameters
        ----------
        initial_iterate : instance of iterate class
            the initial iterate
        log_debug_info : bool, optional
            wether to store and return per-iteration debug information (costs and step sizes). Default: False.
        log_iterates : bool, optional
            wether to store and return all iterates. Default: False.

        Returns
        -------
        final_iterate : backend.array_type 
            the final iterate
        num_iters : int
            the number of iterations the algorithm was run for
        num_restarts_not_descent : int
            number of restarts due to the search direction not being a descent direction
        num_restarts_powell : int
            number of restarts due to Powell's restart strategy (see [5])
        debug_info : tuple
            tuple with debug information, containing three lists: A list of costs (float), a list of step sizes (float),
            and list of iterates (backend.array_type). If log_debug_info == False or log_iterates == False, the corresponding
            entries in the tuple are None.
        """
        # Initital values
        iterate = initial_iterate
        cost = iterate.evaluate_cost_function()
        gradient = iterate.compute_gradient()
        # Iinitialize search direction with negative gradient
        search_direction = -gradient
        # Keep track of the number of CG restarts
        num_restarts_not_descent = 0
        num_restarts_powell = 0
        # Debug logging
        costs = None
        step_sizes = None
        iterates = None
        if log_debug_info:
            costs = [cost]
            step_sizes = []
        if log_iterates:
            iterates = [initial_iterate.get_iterate()]
        if self.manifold.norm(gradient) < self.grad_norm_eps:
            # Termination because of small gradient
            return iterate.get_iterate(), 0, num_restarts_not_descent, num_restarts_powell, (costs, step_sizes, iterates)
        old_alpha = None
        # Main loop
        n = 0
        for _ in range(self.N_iters):
            n += 1
            slope = self.manifold.inner_product(gradient, search_direction)
            if slope >= 0:
                # This is not a descent direction. Restart CG by setting the update direction to the negative gradient
                search_direction = -gradient
                slope = self.manifold.inner_product(gradient, search_direction)
                num_restarts_not_descent += 1
            # Execute line search along update direction
            step_size, new_iterate, new_cost, old_alpha = self._line_search_adaptive(iterate, search_direction, slope, cost, old_alpha)
            if step_size < self.step_size_eps:
                # Termination because of small step_size
                iterate = new_iterate
                cost = new_cost
                if log_debug_info:
                    costs.append(cost)
                    step_sizes.append(step_size)
                if log_iterates:
                    iterates.append(iterate.get_iterate())
                break
            # Compute new gradient and beta
            new_gradient = new_iterate.compute_gradient()
            if self.manifold.norm(new_gradient) < self.grad_norm_eps:
                # Termination because of small gradient
                iterate = new_iterate
                cost = new_cost
                if log_debug_info:
                    costs.append(cost)
                    step_sizes.append(step_size)
                if log_iterates:
                    iterates.append(iterate.get_iterate())
                break
            # Check if we want to restart (Powell's restart strategy, see [5])
            transported_gradient = self.manifold.transport(new_iterate.get_iterate(), gradient)
            beta = 0
            if backend.abs(self.manifold.inner_product(new_gradient, transported_gradient)) >= self.restart_factor * self.manifold.norm(new_gradient) * self.manifold.norm(transported_gradient):
                # Restart CG by setting the update direction to the negative gradient (equivalent to beta = 0)
                num_restarts_powell += 1
            else:
                beta = self.compute_beta(self.manifold, iterate.get_iterate(), new_iterate.get_iterate(), gradient, new_gradient, search_direction)
            # Compute next search direction
            if beta != 0:
                transported_search_direction = self.manifold.transport(new_iterate.get_iterate(), search_direction)
                search_direction = -new_gradient + beta * transported_search_direction
            else:
                search_direction = -new_gradient
            # Update everything
            gradient = new_gradient
            iterate = new_iterate
            cost = new_cost
            if log_debug_info:
                costs.append(cost)
                step_sizes.append(step_size)
            if log_iterates:
                iterates.append(iterate.get_iterate())
        return iterate.get_iterate(), n, num_restarts_not_descent, num_restarts_powell, (costs, step_sizes, iterates)