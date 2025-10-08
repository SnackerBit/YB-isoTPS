from .. import backend
import numpy as np

"""
This file implements the Trust Region method for the optimization on Riemannian manifolds. It uses truncated Conjugate Gradients (tCG)
for solving the trust-region subproblem, as suggested in [1] and [3]
Sources:
[1] P.-A. Absil, Robert Mahony, Rodolphe Sepulchre: "Optimization Algorithms on Matrix Manifolds", https://press.princeton.edu/absil
[2] Markus Hauru, Maarten Van Damme, Jutho Haegeman: "Riemannian optimization of isometric tensor networks", https://scipost.org/10.21468/SciPostPhys.10.2.040
[3] James Townsend, Niklas Koep, Sebastian Weichwald: "Pymanopt: A Python Toolbox for Optimization on Manifolds using Automatic Differentiation", https://arxiv.org/abs/1603.03236
"""

class TrustRegionOptimizer:
    """
    Class implementing the Trust Region method for the optimization on Riemannian manifolds. To use this class, first initialize an instance of
    the class with the desired parameters, and then call optimize().
    The iterates are instances of a iterate class, that is responsible for evaluating the cost function and computing gradients.
    For an example see the RenyiAlphaIterate class from src/utility/disentangle/disentangle_renyi_alpha_trm.py.
    """

    def __init__(self, manifold, construct_iterate, theta_stop=1.0, kappa_stop=0.1, min_inner=3, rho_prime=0.1, rho_regularization=1e3, N_iters=1000, 
                 N_iters_tCG=100, min_gradient_norm=1e-9, min_Delta=1e-9, min_relative_cost_diff=1e-8, Delta0=1.0, Delta_max=1000.0):
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
            The iterate class must implement functions for computing the value of the cost function, the gradient and hessian vector products.
        theta_stop : float, optional
            parameter used in the stopping criterion for TCG to achieve superlinear convergence, see [1]. Must be larger than 0. Default: 1.0.
        kappa_stop : float, optional
            parameter used in the stopping criterion for TCG to achieve superlinear convergence, see [1]. Must be larger than 0. Default: 0.1.
        min_inner : int, optional
            The minimum number of inner TCG iterations done before the inner iteration is terminated. Default: 3.
        rho_prime : float, optional
            comparison parameter 0 <= rho_prime < 1 for deciding if a given iterate is accepted or rejected. Larger rho_prime
            makes it harder for iterates to be accepted. Default: 0.1.
        rho_regularization : float, optional
            regularization parameter for heuristic from Gould and Toint, taken from [3]. Default: 1e3.
        N_iters : int, optional
            The maximum number of iterations the TRM is run for. Default: 1000
        N_iters_tCG : int, optional
            The maximum number of iterations TCG is run to solve the trust-region subproblem. Default: 100.
        min_gradient_norm : float, optional
            If the norm of the gradient is smaller than this threshhold value, the algorithm is terminated. Default: 1e-9.
        min_Delta : float, optional
            If the trust region radius Delta is smaller than this threshhold value, the algorithm is terminated. Default: 1e-9
        min_relative_cost_diff : float, optional
            If the relative cost difference of two iterations is smaller than this threshhold the algorithm is terminated. Default: 1e-8.
        Delta0 : float, optional
            Initial trust region radius. Default: 1.0.
        Delta_max : float, optional
            Maximum allowed trust region radius. Default: 1000.0
        """
        self.manifold = manifold
        self.construct_iterate = construct_iterate
        self.theta_stop = theta_stop
        self.kappa_stop = kappa_stop
        self.min_inner = min_inner
        self.rho_prime = rho_prime
        self.rho_regularization = rho_regularization
        self.N_iters = N_iters
        self.N_iters_tCG = N_iters_tCG
        self.min_gradient_norm = min_gradient_norm
        self.min_Delta = min_Delta
        self.min_relative_cost_diff = min_relative_cost_diff
        self.Delta0 = Delta0
        self.Delta_max = Delta_max

    def _truncated_conjugate_gradients(self, iterate, gradient, Delta):
        """
        Performs the truncated Conjugate Gradients algorithms to find the minimum of the quadratic model of the cost function
        around the current iterate, in the trust region defined by the trust region radius Delta. This approximately solves 
        the trust-region subproblem.

        Parameters
        ----------
        iterate : instance of iterate class
            the current iterate.
        gradient : backend.array_type
            the gradient of the cost function at the current iterate
        Delta : float
            the trust-region radius

        Returns
        -------
        eta : backend.array_type
            the tangent vector approximately solving the trust-region subproblem
        Heta : backend.array_type
            hessian vector product of eta
        N_iters_tCG : int
            number of iterations the subproblem solver was run for        
        hit_delta : bool
            wether the solver hit the border of the trust-region. If the border was not hit,
            either the minimum of the model is inside the trust region or tCG did not converge yet.
        """
        # The goal is to compute update vector eta. We start with the zero vector
        eta = self.manifold.zero_vector()
        eta_eta = self.manifold.inner_product(eta, eta) # TODO: Is this necessary? eta is the zero vector ...
        # Initial search direction
        r = gradient
        r_r = self.manifold.inner_product(r, r)
        norm_r_0 = backend.sqrt(r_r) # necessary for resiudal stopping criterion
        delta = -r
        delta_delta = self.manifold.inner_product(delta, delta)
        eta_delta = self.manifold.inner_product(eta, delta) # TODO: Is this necessary? eta is the zero vector ...
        Heta = iterate.compute_hessian_vector_product(eta) # TODO: Is this necessary? eta is the zero vector ...
        if delta_delta == 0 or r_r == 0:
            return eta, Heta, 0, False
        # Wether we have hit the TR radius
        hit_delta = False

        # function for evaluating the model
        def _evaluate_model(eta, Heta):
            return self.manifold.inner_product(gradient, eta) + 0.5 * self.manifold.inner_product(Heta, eta)

        model_value = 0
        N_iters_tCG = self.N_iters_tCG

        for j in range(self.N_iters_tCG):
            # Compute the hessian vector product of the current search direction. This is expensive ...
            Hdelta = iterate.compute_hessian_vector_product(delta)
            # Compute curvature
            kappa = self.manifold.inner_product(delta, Hdelta)
            # Check if the curvature is negative
            if kappa <= 0 or backend.isnan(kappa) or backend.isinf(kappa):
                # Curvature is negative -> Go as far as we can in the current update direction, ie. to the trust-region border.
                # Thus, we need to find tau such that ||eta + tau * delta|| = Delta. We can find such a tau by solving
                # for the positive root of a simple quadratic equation
                tau = (eta_delta + backend.sqrt(eta_delta**2 + delta_delta * (Delta**2 - eta_eta))) / delta_delta
                # Update eta
                eta = eta + tau * delta
                # Compute new eta. This is cheap, but only an approximation if the approximate Hessian is non-linear!
                Heta = Heta + tau * Hdelta
                hit_delta = True
                break
            # Curvature is not negative -> find minimum along search direction using exact line search!
            # This is similar to classic conjugate gradients!
            alpha = r_r / kappa
            new_eta = eta + alpha * delta
            new_eta_eta = eta_eta + 2*alpha*eta_delta + alpha**2*delta_delta

            # Check if we stepped outside of the trust-region
            if new_eta_eta >= Delta**2:
                # Go as far as we can in the current update direction, ie. to the trust-region border.
                # Thus, we need to find tau such that ||eta + tau * delta|| = Delta. We can find such a tau by solving
                # for the positive root of a simple quadratic equation
                tau = (eta_delta + backend.sqrt(eta_delta**2 + delta_delta * (Delta**2 - eta_eta))) / delta_delta
                # Update eta
                eta = eta + tau * delta
                # Compute new eta. This is cheap, but only an approximation if the approximate Hessian is non-linear!
                Heta = Heta + tau * Hdelta
                hit_delta = True
                N_iters_tCG = j+1
                break
            # We are still inside the trust region, and thus accept the new eta. 
            # Recompute the approximate hessian vector product of eta. Instead of calling the full computation of the hvp, we can apply a simple update.
            # This update is cheaper but is only an approximation if the hessian approximation is non-linear
            new_Heta = Heta + alpha * Hdelta
            # Check if the model value has increased
            new_model_value = _evaluate_model(new_eta, new_Heta)
            if new_model_value >= model_value:
                # model value has not decreased. This can occur if the hessian approximation is non-linear,
                # or due to numerical errors. In this case, we just return the best iterate so far, which is eta
                N_iters_tCG = j+1
                break
            # model value has decreased. Accept new eta
            eta = new_eta
            eta_eta = new_eta_eta
            Heta = new_Heta
            model_value = new_model_value
            # Compute the next residual
            r = r + alpha * Hdelta
            r_r_old = r_r
            r_r = self.manifold.inner_product(r, r)
            if r_r == 0.0:
                N_iters_tCG = j+1
                break
            norm_r = backend.sqrt(r_r)
            # Check the theta/kappa stopping criterion from the book, in order to achieve superlinear convergence
            if j >= self.min_inner and r_r <= norm_r_0 * min(norm_r_0**self.theta_stop, self.kappa_stop):
                # resiudal is small enough, and we can quit
                N_iters_tCG = j+1
                break
            # Compute new search direction
            beta = r_r / r_r_old
            delta = -r + beta * delta
            # Make sure that update vector remains in tangent space (can diverge due to numerical errors)
            delta = self.manifold.project_to_tangent_space(iterate.get_iterate(), delta)
            # Recompute helper variables
            #delta_delta = self.manifold.inner_product(delta, delta)
            eta_delta = beta * (eta_delta + alpha * delta_delta)
            #eta_delta = self.manifold.inner_product(eta, delta)
            delta_delta = r_r + beta**2 * delta_delta
        return eta, Heta, N_iters_tCG, hit_delta

    def optimize(self, initial_iterate, log_debug_info=False, log_iterates=False, print_warnings=False):
        """
        Optimizes the given initial iterate with the Trust Region method.

        Parameters
        ----------
        initial_iterate : instance of iterate class
            the initial iterate
        log_debug_info : bool, optional
            wether to store and return per-iteration debug information (costs, trust region radii and tCG iterations). Default: False.
        log_iterates : bool, optional
            wether to store and return all iterates. Default: False.
        print_warnings : bool, optional
            wether to print warnings when many consecutive trust-region radius increases or decreases occur. Default : False.

        Returns
        -------
        final_iterate : backend.array_type 
            the final iterate
        num_iters : int
            the number of iterations the algorithm was run for
        debug_info : tuple
            tuple with debug information, containing four lists: A list of costs (float), a list of trust-region radii (float),
            a list of tCG iteration counts (int) and a list of iterates (backend.array_type). If log_debug_info == False or 
            log_iterates == False, the corresponding entries in the tuple are None.
        """
        iterate = initial_iterate
        cost = iterate.evaluate_cost_function()
        gradient = iterate.compute_gradient()
        Delta = self.Delta0
        gradient_norm = self.manifold.norm(gradient)
        costs = None
        Deltas = None
        N_iters_tCG_list = None
        iterates = None
        if log_debug_info:
            costs = [cost]
            Deltas = [Delta]
            N_iters_tCG_list = []
        if log_iterates:
            iterates = [initial_iterate.get_iterate()]
        # relative cost diff of this and last iteration (stopping criterion)
        relative_cost_diff = None
        # Helper variables to keep track of consecutive trust region radius changes. We may want to warn the user if the radius changes a lot!
        if print_warnings:
            consecutive_Delta_changes_plus = 0
            consecutive_Delta_changes_minus = 0
        # Start the TRM main loop
        n = 0
        for _ in range(self.N_iters):
            n += 1
            # Approximately solve trust-region subproblem
            eta, Heta, N_iters_tCG, hit_delta = self._truncated_conjugate_gradients(iterate, gradient, Delta)
            if log_debug_info:
                N_iters_tCG_list.append(N_iters_tCG)
            # Compute the next iterate proposed by the approximate solution to the trust-region subproblem
            x_proposed = self.manifold.retract(iterate.get_iterate(), eta)
            proposed_iterate = self.construct_iterate(x_proposed, iterate)
            # Compute the value of the cost function at this proposed iterate
            proposed_cost = proposed_iterate.evaluate_cost_function()
            # Compute rho, which is used to compare the current model to the actual cost function
            rho_numerator = cost - proposed_cost # if the actual cost decreased, this should be positive
            # The subproblem solver guarantees the denominator to be positive, except for numerical errors.
            rho_denominator = -self.manifold.inner_product(gradient, eta) - self.manifold.inner_product(Heta, eta) / 2
            # Apply heuristic from book by onn Gould and Toint
            rho_reg = max(1, abs(cost)) * np.spacing(1) * self.rho_regularization
            rho_numerator += rho_reg
            rho_denominator += rho_reg
            # Check if the model decreased, which can happen due to numerical errors
            model_decreased = rho_denominator >= 0
            
            # Compute rho
            try:
                rho = rho_numerator / rho_denominator
            except ZeroDivisionError:
                rho = backend.nan
                print("[WARNING]: Division by zero occurred in trust-region method. This should not happen.")

            # Now, we choose the new trust-region radius based on the models performance
            # if the actual decrease is smaller than 1/4 of the predicted decrease, then reduce the TR radius!
            if rho < 1.0/4.0 or not model_decreased or backend.isnan(rho):
                Delta = Delta / 4
                if print_warnings:
                    consecutive_Delta_changes_plus = 0
                    consecutive_Delta_changes_minus += 1
                    if consecutive_Delta_changes_minus >= 5:
                        print("[WARNING]: Detected many consecutive TR radius decreases. Consider decreasing Delta0 by an order of magnitude.")
                        consecutive_Delta_changes_minus = -backend.inf
            # If the actual decrease is at least 3/4 of the predicted decrease and the trust-region subproblem solver
            # hit the boundary of the trust-region, increase the TR radius
            elif rho > 3.0/4.0 and hit_delta:
                Delta = min(2*Delta, self.Delta_max)
                if print_warnings:
                    consecutive_Delta_changes_plus += 1
                    consecutive_Delta_changes_minus = 0
                    if consecutive_Delta_changes_plus >= 5:
                        print("[WARNING]: Detected many consecutive TR radius increases. Consider increasing Delta_max by an order of magnitude.")
                        consecutive_Delta_changes_minus = -backend.inf
            # Otherwise we keep the TR radius constant
            else:
                if print_warnings:
                    consecutive_Delta_changes_plus = 0
                    consecutive_Delta_changes_minus = 0

            # Next we choose wether to accept or reject the proposed step based on the models performance
            if model_decreased and rho > self.rho_prime:
                # accept the proposed update
                iterate = proposed_iterate
                if not backend.isclose(cost, 0):
                    relative_cost_diff = (cost - proposed_cost) / backend.abs(cost)
                else:
                    relative_cost_diff = None
                cost = proposed_cost
                gradient = iterate.compute_gradient()
                gradient_norm = self.manifold.norm(gradient)
            else:
                # reject the proposed update
                relative_cost_diff = None
                
            if log_debug_info:
                costs.append(cost)
                Deltas.append(Delta)
            if log_iterates:
                iterates.append(iterate.get_iterate())
            
            # Check for stopping criterion
            if gradient_norm < self.min_gradient_norm or Delta < self.min_Delta or (relative_cost_diff is not None and backend.abs(relative_cost_diff) < self.min_relative_cost_diff):
                break

        return iterate.get_iterate(), n, (costs, Deltas, N_iters_tCG_list, iterates)