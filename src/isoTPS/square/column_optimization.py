import numpy as np
from ...utility import utility
from ...utility import debug_logging
from .. import shifting_ortho_center

class variationalColumnOptimizer:

    def __init__(self, Ts_before_YB, Ws_before_YB, Ts_after_YB, Ws_after_YB, ortho_center, chi_max, mode="dont_move_oc", relative_improvement_stopping_criterion=1e-9, debug_logger=debug_logging.DebugLogger()):
        self.Ts_before_YB = Ts_before_YB
        self.Ws_before_YB = Ws_before_YB
        self.Ts = Ts_after_YB
        self.Ws = Ws_after_YB
        self.ortho_center = ortho_center
        self.chi_max = chi_max
        self.mode = mode
        self.relative_improvement_stopping_criterion = relative_improvement_stopping_criterion
        self.debug_logger = debug_logger
        self.Ly = len(self.Ts_before_YB)
        assert(len(self.Ws_before_YB) == 2*self.Ly)
        assert(len(self.Ts) == self.Ly)
        assert(len(self.Ws) == 2*self.Ly)
        self.Es_bot = [None for i in range(self.Ly)]
        self.Es_bot[0] = np.ones((1, 1), dtype=complex)
        self.Es_top = [None for i in range(self.Ly)]
        self.Es_top[-1] = np.ones((1, 1), dtype=complex)
        self.thetas = [None for i in range(self.Ly)]
        self.thetaTs = [None for i in range(self.Ly)]
        self.W1W2s = [None for i in range(self.Ly)]
        self.eps = None
        self.sanity_check()

    def sanity_check(self):
        # sanity check
        for n in range(self.ortho_center):
            #if self.Ws_before_YB[n] is not None:
            #    temp = np.transpose(self.Ws_before_YB[n], (0, 2, 3, 1)) # l, u, r, d -> l, r, d, u
            #    l, r, d, u = temp.shape
            #    temp = np.reshape(temp, (l*r*d, u)) # l, r, d, u -> (l, r, d), u
            #    assert utility.check_isometry(temp), f"Ws_before_YB[{n}] is not an isometry! (ortho_center at {self.ortho_center})"
            if self.Ws[n] is not None:
                temp = np.transpose(self.Ws[n], (0, 2, 3, 1)) # l, u, r, d -> l, r, d, u
                l, r, d, u = temp.shape
                temp = np.reshape(temp, (l*r*d, u)) # l, r, d, u -> (l, r, d), u
                assert utility.check_isometry(temp), f"W[{n}] is not an isometry! (ortho_center at {self.ortho_center})"
        for n in range(self.ortho_center+1, len(self.Ws_before_YB)):
            #if self.Ws_before_YB[n] is not None:
            #    l, u, r, d = self.Ws_before_YB[n].shape
            #    temp = np.reshape(self.Ws_before_YB[n], (l*u*r, d)) # l, u, r, d -> (l, u, r), d
            #    assert utility.check_isometry(temp), f"Ws_before_YB[{n}] is not an isometry! (ortho_center at {self.ortho_center})"
            if self.Ws[n] is not None:
                l, u, r, d = self.Ws[n].shape
                temp = np.reshape(self.Ws[n], (l*u*r, d)) # l, u, r, d -> (l, u, r), d
                assert utility.check_isometry(temp), f"Ws[{n}] is not an isometry! (ortho_center at {self.ortho_center})"
        #assert np.isclose(np.linalg.norm(self.Ws_before_YB[self.ortho_center]), 1), f"Ws_before_YB[{n}] is not a tensor of norm one! (ortho_center at {self.ortho_center})"
        assert np.isclose(np.linalg.norm(self.Ws[self.ortho_center]), 1), f"Ws[{n}] is not a tensor of norm one! (ortho_center at {self.ortho_center})"

    @staticmethod
    def contract_theta(W1, W2, T):
        r"""
        Contracts W1, W2 and T into a single tensor at a cost of O(chi^3D^4 + chi^2D^6d) = O(D^8)

            \ |      
             \|      
             W2      
             |\   / 
             | \ /  
             |  T-- 
             | / \  
             |/   \ 
             W1      
             /|      
            / |      

        """
        if W1 is None:
            contr = np.tensordot(W2, T, ([2], [4])) # l u [r] d; p ru rd ld [lu] -> l u d p ru rd ld
            contr = np.transpose(contr, (6, 2, 0, 1, 3, 4, 5)) # l u d p ru rd ld = l2 u2 d1 p ru rd l1 -> l1 d1 l2 u2 p ru rd
            return contr
        elif W2 is None:
            contr = np.tensordot(W1, T, ([2], [3])) # l u [r] d; p ru rd [ld] lu -> l u d p ru rd lu
            contr = np.transpose(contr, (0, 2, 6, 1, 3, 4, 5)) # l u d p ru rd lu = l1 u2 d1 p ru rd l2 -> l1 d1 l2 u2 p ru rd
            return contr
        else:
            contr = np.tensordot(W1, W2, ([1], [3])) # l1 [u1] r1 d1; l2 u2 r2 [d2] -> l1 r1 d1 l2 u2 r2
            contr = np.tensordot(contr, T, ([1, 5], [3, 4])) # l1 [r1] d1 l2 u2 [r2]; p ru rd [ld] [lu] -> l1 d1 l2 u2 p ru rd
            return contr

    def notify_changed_W(self, W_index):
        T_index = W_index//2
        if T_index > 0:
            self.Es_top[T_index-1] = None
        if T_index < self.Ly-1:
            self.Es_bot[T_index+1] = None
        self.W1W2s[T_index] = None

    def compute_thetas(self):
        if self.thetas[0] is not None:
            return
        for i in range(self.Ly):
            self.thetas[i] = variationalColumnOptimizer.contract_theta(self.Ws_before_YB[2*i], self.Ws_before_YB[2*i+1], self.Ts_before_YB[i])

    def compute_thetaT(self, i): 
        """
        Contracts theta with T at position i, assuming that the thetas are already computed
        """
        self.thetaTs[i] = np.tensordot(self.thetas[i], np.conj(self.Ts[i]), ([0, 2, 4], [3, 4, 0])) # [l1] d1 [l2] u2 [p] ru rd; [p*] ru* rd* [ld*] [lu*] -> d1 u2 ru rd ru* rd*

    def compute_W1W2(self, i):
        """
        Computes the contraction of W1 and W2 at position i, assuming that neither W1 nor W2 are None
        """
        self.W1W2s[i] = np.conj(np.tensordot(self.Ws[2*i], self.Ws[2*i+1], ([1], [3]))) # l1 [u1] r1 d1; l2 u2 r2 [d2] -> l1 r1 d1 l2 u2 r2
        
    def contract_bottom_environment(self, i):
        r"""
        Computes the next bottom environment E' as

                                 _|_      |  
                                |   |____ |  
                                |   |    \|  
                                |   |    W2* 
                                |   |\   /|  
                                |   | \ / |  
             _|_______|_        |th |--T  |  
            |_____E'____|   =   |   | / \ |  
                                |   |/   \|  
                                |   |     W1*
                                |   |____/|  
                                |___|     |  
                                 _|_______|_ 
                                |_____E_____|

        """
        if self.thetaTs[i] is None:
            self.compute_thetaT(i)
        contr = np.tensordot(self.Es_bot[i], self.thetaTs[i], ([0], [0])) # [e] e*; [d1] u2 ru rd ru* rd* -> e* u2 ru rd ru* rd*
        if self.Ws[2*i] is None:
            contr = np.tensordot(contr[:, :, :, 0, :, 0], np.conj(self.Ws[2*i+1]), ([0, 2, 3], [3, 2, 0])) # [e*] u2 [ru] [ru*]; [l2*] u2* [r2*] [d2*] -> u2 u2* = e e*
        elif self.Ws[2*i+1] is None:
            #print(contr.shape, self.Ws[2*i].shape)
            #print([e.shape for e in self.Es_bot])
            #print([None if W is None else W.shape for W in self.Ws])
            contr = np.tensordot(contr[:, :, 0, :, 0, :], np.conj(self.Ws[2*i]), ([0, 2, 3], [3, 2, 0])) # [e*] u2 [rd] [rd*]; [l1*] u1* [r1*] [d1*] -> u2 u1* = e e*
        else:
            if self.W1W2s[i] is None:
                self.compute_W1W2(i)
            contr = np.tensordot(contr, self.W1W2s[i], ([0, 2, 3, 4, 5], [2, 5, 1, 3, 0])) # [e*] u2 [ru] [rd] [ru*] [rd*]; [l1*] [r1*] [d1*] [l2*] u2* [r2*] -> u2 u2* = e e*
        if i == self.Ly - 1:
            # Compute error
            temp = 2 - 2*np.real(contr.item())
            if np.isclose(temp, 0):
                temp = 0.0
            self.eps = np.sqrt(temp)
        else:
            self.Es_bot[i+1] = contr

    def contract_top_environment(self, i):
        r"""
        Computes the top next environment E' as

                                 ___________ 
                                |_____E_____|
                                 _|_      |  
                                |   |____ |  
                                |   |    \|  
                                |   |    W2* 
                                |   |\   /|  
                                |   | \ / |  
             ___________        |th |--T  |  
            |_____E'____|   =   |   | / \ |  
              |       |         |   |/   \|  
                                |   |     W1*
                                |   |____/|  
                                |___|     |  
                                  |       |  

        """
        if self.thetaTs[i] is None:
            self.compute_thetaT(i)
        contr = np.tensordot(self.Es_top[i], self.thetaTs[i], ([0], [1])) # [e] e*; d1 [u2] ru rd ru* rd* -> e* d1 ru rd ru* rd*
        if self.Ws[2*i] is None:
            contr = np.tensordot(contr[:, :, :, 0, :, 0], np.conj(self.Ws[2*i+1]), ([0, 2, 3], [1, 2, 0])) # [e*] d1 [ru] [ru*]; [l2*] [u2*] [r2*] d2* -> d1 d2* = e e*
        elif self.Ws[2*i+1] is None:
            contr = np.tensordot(contr[:, :, 0, :, 0, :], np.conj(self.Ws[2*i]), ([0, 2, 3], [1, 2, 0])) # [e*] d1 [rd] [rd*]; [l1*] [u1*] [r1*] d1* -> d1 d1* = e e*
        else:
            if self.W1W2s[i] is None:
                self.compute_W1W2(i)
            contr = np.tensordot(contr, self.W1W2s[i], ([0, 2, 3, 4, 5], [4, 5, 1, 3, 0])) # [e*] d1 [ru] [rd] [ru*] [rd*]; [l1*] [r1*] d1* [l2*] [u2*] [r2*] -> d1 d1* = e e*
        if i == 0:
            # Compute error
            temp = 2 - 2*np.real(contr.item())
            if np.isclose(temp, 0):
                temp = 0.0
            self.eps = np.sqrt(temp)
        else:
            self.Es_top[i-1] = contr

    def compute_bottom_environments(self):
        """
        Computes bottom environments starting from the bottom of the column and moving upwards
        """
        self.Es_bot[0] = np.ones((1, 1), dtype=complex)
        for i in range(self.Ly):
            #print(f"Computing bottom env {i}")
            self.contract_bottom_environment(i)

    def compute_top_environments(self):
        """
        Computes top environments starting from the top of the column and moving downwards
        """
        self.Es_top[-1] = np.ones((1, 1), dtype=complex)
        for i in range(self.Ly-1, -1, -1):
            self.contract_top_environment(i)

    def compute_error(self):
        """
        Computes the error by contracting all environments
        """
        # Initial contractions
        if self.thetas[0] is None:
            self.compute_thetas()
        # Compute environment and error
        self.compute_bottom_environments()
        # return error
        return self.eps

    def move_ortho_center_up(self):
        """
        Moves the ortho center on the Ws_after_YB one spot upwards.
        """
        #print(f"Moving OC up from {self.ortho_center}")
        N = len(self.Ws)
        if self.ortho_center == N-1 or self.ortho_center == N-2 and self.Ws[N-1] is None:
            return
        self.Ws[self.ortho_center], self.Ws[self.ortho_center+1] = shifting_ortho_center.move_ortho_center_up(self.Ws[self.ortho_center], self.Ws[self.ortho_center+1], self.chi_max)
        self.notify_changed_W(self.ortho_center)
        self.notify_changed_W(self.ortho_center+1)   
        self.ortho_center += 1

    def move_ortho_center_down(self):
        """
        Moves the ortho center on the Ws_after_YB one spot downwards.
        """
        #print(f"Moving OC up down {self.ortho_center}")
        if self.ortho_center == 0 or self.ortho_center == 1 and self.Ws[0] is None:
            return
        self.Ws[self.ortho_center-1], self.Ws[self.ortho_center] = shifting_ortho_center.move_ortho_center_down(self.Ws[self.ortho_center-1], self.Ws[self.ortho_center], self.chi_max)
        self.W1W2s[self.ortho_center//2] = None
        self.notify_changed_W(self.ortho_center)
        self.notify_changed_W(self.ortho_center-1)   
        self.ortho_center -= 1

    def move_ortho_center_to(self, i):
        """
        Moves the ortho center to the specified location
        """
        i = max(min(i, 2*self.Ly-1), 0)
        if i == 0 and self.Ws[0] is None:
            i = 1
        elif i == 2*self.Ly-1 and self.Ws[2*self.Ly-1] is None:
            i = 2*self.Ly-2
        if self.ortho_center < i:
            while self.ortho_center != i:
                self.move_ortho_center_up()
        else:
            while self.ortho_center != i:
                self.move_ortho_center_down()

    def optimize_T(self, i):
        """
        variationally optimizes the T tensor at the given index. Assumes that the top and bottom environments at i
        exist and are up to date and that neither W1 nor W2 are None. This function reuses stored contractions if possible.
        """
        #print(f"Optimizing T at {i}")
        #self.print_environments()
        assert self.Es_bot[i] is not None
        assert self.Es_top[i] is not None
        # Contract theta with environments
        contr = np.tensordot(self.thetas[i], self.Es_bot[i], ([1], [0])) # l1 [d1] l2 u2 p ru rd; [eb] eb* -> l1 l2 u2 p ru rd eb*
        contr = np.tensordot(contr, self.Es_top[i], ([2], [0])) # l1 l2 [u2] p ru rd eb*; [et] et* -> l1 l2 p ru rd eb* et*
        # Contract with W1 and W2
        if self.W1W2s[i] is None:
            self.compute_W1W2(i)
        contr = np.tensordot(contr, self.W1W2s[i], ([3, 4, 5, 6], [5, 1, 2, 4])) # l1 l2 p [ru] [rd] [eb*] [et*]; l1* [r1*] [d1*] l2* [u2*] [r2*] -> l1 l2 p l1* l2* = ld lu p rd ru
        # Isometrize and transpose
        ld, lu, p, rd, ru = contr.shape # l1 l2 p l1* l2* = ld lu p rd ru
        contr = np.reshape(contr, (ld*lu*p, rd*ru)) # ld, lu, p, rd, ru -> (ld, lu, p), (rd, ru)
        contr = utility.isometrize_polar(contr)
        contr = np.reshape(contr, (ld, lu, p, rd, ru)) # (ld, lu, p), (rd, ru) -> d, lu, p, rd, ru
        contr = np.transpose(contr, (2, 4, 3, 0, 1)) # ld, lu, p, rd, ru -> p, ru, rd, ld, lu
        # Store the resulting tensor
        self.Ts[i] = contr
        self.thetaTs[i] = None
        if i > 0:
            self.Es_top[i-1] = None
        if i < self.Ly-1:
            self.Es_bot[i+1] = None

    def optimize_W1(self, i):
        """
        variationally optimizes the W1 tensor at the given index. Assumes that the top and bottom environments at i
        exist and are up to date and that neither W1 nor W2 are None. This function reuses stored contractions if possible.
        """
        #print(f"Optimizing W1 at {i}")
        #self.print_environments()
        assert self.Es_bot[i] is not None
        assert self.Es_top[i] is not None
        if self.mode == "move_oc_first_T_then_W":
            assert self.ortho_center == 2*i
        # Contract thetaT
        if self.thetaTs[i] is None:
            self.compute_thetaT(i)
        # Contract with top environment
        contr = np.tensordot(self.thetaTs[i], self.Es_top[i], ([1], [0])) # d1 [u2] ru rd ru* rd*; [et] et* -> d1 ru rd ru* rd* et*
        # Contract with W2
        contr = np.tensordot(contr, np.conj(self.Ws[2*i+1]), ([1, 3, 5], [2, 0, 1])) # d1 [ru] rd [ru*] rd* [et*]; [l2*] [u2*] [r2*] d2* -> d1 rd rd* d2*
        if 2*i < self.ortho_center:
            # Contract with bottom environment
            contr = np.tensordot(self.Es_bot[i], contr, ([0], [0])) # [eb] eb*; [d1] rd rd* d2* -> eb* rd rd* d2 = d1 r1 l1 u1
            # Isometrize and transpose
            d1, r1, l1, u1 = contr.shape # d1, r1, l1, u1
            contr = np.reshape(contr, (d1*r1*l1, u1)) # d1, r1, l1, u1 -> (d1, r1, l1), u1
            contr = utility.isometrize_polar(contr)
            contr = np.reshape(contr, (d1, r1, l1, u1)) # (d1, r1, l1), u1 -> d1, r1, l1, u1
            contr = np.transpose(contr, (2, 3, 1, 0)) # d1, r1, l1, u1 -> l1, u1, r1, d1
        else:
            # Contract with bottom environment
            contr = np.tensordot(contr, self.Es_bot[i], ([0], [0])) # [d1] rd rd* d2*; [eb] eb* -> rd rd* d2 eb* = r1 l1 u1 d1
            if 2*i > self.ortho_center:
                # Isometrize and transpose
                r1, l1, u1, d1 = contr.shape # r1, l1, u1, d1
                contr = np.reshape(contr, (r1*l1*u1, d1)) # r1, l1, u1, d1 -> (r1, l1, u1), d1
                contr = utility.isometrize_polar(contr)
                contr = np.reshape(contr, (r1, l1, u1, d1)) # (r1, l1, u1), d1 -> r1, l1, u1, d1
            else:
                # Normalize
                #print("ortho surface is at W1")
                contr /= np.linalg.norm(contr)
            contr = np.transpose(contr, (1, 2, 0, 3)) # r1, l1, u1, d1 -> l1, u1, r1, d1
        # Store the resulting tensor
        self.Ws[2*i] = contr
        self.notify_changed_W(2*i)

    def optimize_W2(self, i):
        """
        Variationally optimizes the W2 tensor at the given index. Assumes that the top and bottom environments at i
        exist and are up to date and that neither W1 nor W2 are None. This function reuses stored contractions if possible.
        """
        #print(f"Optimizing W2 at {i}")
        #self.print_environments()
        assert self.Es_bot[i] is not None
        assert self.Es_top[i] is not None
        if self.mode == "move_oc_first_T_then_W":
            assert self.ortho_center == 2*i+1
        # Contract thetaT
        if self.thetaTs[i] is None:
            self.compute_thetaT(i)
        # Contract with bottom environment
        contr = np.tensordot(self.thetaTs[i], self.Es_bot[i], ([0], [0])) # [d1] u2 ru rd ru* rd*; [eb] eb* -> u2 ru rd ru* rd* eb*
        # Contract with W1
        contr = np.tensordot(contr, np.conj(self.Ws[2*i]), ([2, 4, 5], [2, 0, 3])) # u2 ru [rd] ru* [rd*] [eb*]; [l1*] u1* [r1*] [d1*] -> u2 ru ru* u1*
        if 2*i+1 < self.ortho_center:
            # Contract with top environment
            contr = np.tensordot(contr, self.Es_top[i], ([0], [0])) # [u2] ru ru* u1*; [et] et* -> ru ru* u1* et* = r2 l2 d2 u2
            # Isometrize and transpose
            r2, l2, d2, u2 = contr.shape # r2, l2, d2, u2
            contr = np.reshape(contr, (r2*l2*d2, u2)) # r2, l2, d2, u2 -> (r2, l2, d2), u2
            contr = utility.isometrize_polar(contr)
            contr = np.reshape(contr, (r2, l2, d2, u2)) # (r2, l2, d2), u2 -> r2, l2, d2, u2
            contr = np.transpose(contr, (1, 3, 0, 2)) # r2, l2, d2, u2 -> l2, u2, r2, d2
        else:
            # Contract with top environment
            contr = np.tensordot(self.Es_top[i], contr, ([0], [0])) # [et] et*; [u2] ru ru* u1* -> et* ru ru* u1* = u2 r2 l2 d2
            if 2*i+1 > self.ortho_center:
                # Isometrize and transpose
                u2, r2, l2, d2 = contr.shape # u2, r2, l2, d2 
                contr = np.reshape(contr, (u2*r2*l2, d2)) # u2, r2, l2, d2 -> (u2, r2, l2), d2
                contr = utility.isometrize_polar(contr)
                contr = np.reshape(contr, (u2, r2, l2, d2)) # (u2, r2, l2), d2 -> u2, r2, l2, d2
            else:
                # Normalize
                contr /= np.linalg.norm(contr)
                #print("ortho surface is at W2")
            contr = np.transpose(contr, (2, 0, 1, 3)) # u2, r2, l2, d2 -> l2, u2, r2, d2
        # Store the resulting tensor
        self.Ws[2*i+1] = contr
        self.notify_changed_W(2*i+1)
        
    def optimize_T_W1(self, i):
        """
        Optimizes T and W1 at position i. Assumes that W2 is None and that i is at the top of the column. 
        Assumes that the top and bottom environments at i exist and are up to date. 
        This function reuses stored contractions if possible.
        """
        assert self.Es_bot[i] is not None
        assert self.Es_top[i] is not None
        if self.mode == "move_oc_first_T_then_W":
            assert self.ortho_center == 2*i
        # Contract theta with environments
        Etheta = np.tensordot(self.thetas[i][:, :, 0, :, :, 0, :], self.Es_bot[i], ([1], [0])) # l1 [d1] u2 p rd; [eb] eb* -> l1 u2 p rd eb*
        Etheta = np.tensordot(Etheta, self.Es_top[i], ([1], [0])) # l1 [u2] p rd eb*; [et] et* -> l1 p rd eb* et*
        # Optimize T
        contr = np.tensordot(Etheta, np.conj(self.Ws[2*i]), ([2, 3, 4], [2, 3, 1])) # l1 p [rd] [eb*] [et*]; l1* [u1*] [r1*] [d1*] -> l1 p l1* = ld p rd
        # Isometrize and transpose
        ld, p, rd = contr.shape
        contr = np.reshape(contr, (ld*p, rd)) # ld, p, rd -> (ld, p), rd)
        contr = utility.isometrize_polar(contr)
        contr = np.reshape(contr, (ld, p, rd, 1, 1)) # (ld, p), rd -> ld, p, rd, ru, lu
        contr = np.transpose(contr, (1, 3, 2, 0, 4)) # ld, p, rd, ru, lu -> p, ru, rd, ld, lu
        # Store the resulting tensor
        self.Ts[i] = contr
        self.thetaTs[i] = None
        # Optimize W1
        self.compute_thetaT(i)
        contr = np.tensordot(self.thetaTs[i][:, :, 0, :, 0, :], self.Es_top[i], ([1], [0])) # d1 [u2] rd rd*; [et] et* -> d1 rd rd* et*
        contr = np.tensordot(contr, self.Es_bot[i], ([0], [0])) # [d1] rd rd* et*; [eb] eb* -> rd rd* et* eb* = r1 l1 u1 d1
        if self.ortho_center == 2*i:
            # Normalize
            contr /= np.linalg.norm(contr)
        else:
            # Isometrize
            r1, l1, u1, d1 = contr.shape
            contr = np.reshape(contr, (r1*l1*u1, d1)) # r1, l1, u1, d1 -> (r1, l1, u1), d1
            contr = utility.isometrize_polar(contr)
            contr = np.reshape(contr, (r1, l1, u1, d1)) # (r1, l1, u1), d1 -> r1, l1, u1, d1
        contr = np.transpose(contr, (1, 2, 0, 3)) # r1, l1, u1, d1 -> l1, u1, r1, d1
        # Store the resulting tensor
        self.Ws[2*i] = contr
        self.Es_top[i-1] = None

    def optimize_T_W2(self, i):
        """
        Optimizes T and W1 at position i. Assumes that W1 is None and that i is at the bottom of the column. 
        Assumes that the top and bottom environments at i exist and are up to date. 
        This function reuses stored contractions if possible.
        """
        assert self.Es_bot[i] is not None
        assert self.Es_top[i] is not None
        if self.mode == "move_oc_first_T_then_W":
            assert self.ortho_center == 2*i+1
        # Contract theta with environments
        Etheta = np.tensordot(self.thetas[i][0, :, :, :, :, :, 0], self.Es_bot[i], ([0], [0])) # [d1] l2 u2 p ru; [eb] eb* -> l2 u2 p ru eb*
        Etheta = np.tensordot(Etheta, self.Es_top[i], ([1], [0])) # l2 [u2] p ru eb*; [et] et* -> l2 p ru eb* et*
        # Optimize T
        contr = np.tensordot(Etheta, np.conj(self.Ws[2*i+1]), ([2, 3, 4], [2, 3, 1])) # l2 p [ru] [eb*] [et*]; l2* [u2*] [r2*] [d2*] -> l2 p l2* = lu p ru
        # Isometrize and transpose
        lu, p, ru = contr.shape
        contr = np.reshape(contr, (lu*p, ru)) # lu, p, ru -> (lu, p), ru
        contr = utility.isometrize_polar(contr)
        contr = np.reshape(contr, (lu, p, ru, 1, 1)) # (lu, p), ru -> lu, p, ru, rd, ld
        contr = np.transpose(contr, (1, 2, 3, 4, 0)) # lu, p, ru, rd, ld -> p, ru, rd, ld, lu
        # Store the resulting tensor
        self.Ts[i] = contr
        self.thetaTs[i] = None
        # Optimize W2
        self.compute_thetaT(i)
        contr = np.tensordot(self.thetaTs[i][:, :, :, 0, :, 0], self.Es_bot[i], ([0], [0])) # [d1] u2 ru ru*; [eb] eb* -> u2 ru ru* eb*
        contr = np.tensordot(contr, self.Es_top[i], ([0], [0])) # [u2] ru ru* eb*; [et] et* -> ru ru* eb* et* = r2 l2 d2 u2
        if self.ortho_center == 2*i+1:
            # Normalize
            contr /= np.linalg.norm(contr)
        else:
            # Isometrize
            r2, l2, d2, u2 = contr.shape
            contr = np.reshape(contr, (r2*l2*d2, u2)) # r2, l2, d2, u2 -> (r2, l2, d2), u2
            contr = utility.isometrize_polar(contr)
            contr = np.reshape(contr, (r2, l2, d2, u2)) #  (r2, l2, d2), u2 -> r2, l2, d2, u2
        contr = np.transpose(contr, (1, 3, 0, 2)) # r2, l2, d2, u2 -> l2, u2, r2, d2
        # Store the resulting tensor
        self.Ws[2*i+1] = contr
        self.Es_bot[i+1] = None

    def print_environments(self):
        # DEBUG
        print("bottom environments:", [False if E is None else True for E in self.Es_bot])
        print("top environments:", [False if E is None else True for E in self.Es_top])

    def optimize_column_sweep_bottom_to_top(self):
        """
        Sweeps once from the bottom to the top, optimizing T and W tensors and updating the bottom environments along the way.
        The bottom most tensors (position 0) are not updated. Assumes all thetas have been computed already.
        """
        #print("Column sweep bottom to top")
        if self.Es_bot[1] is None:
            self.contract_bottom_environment(0)
        for i in range(1, self.Ly-1):
            self.optimize_T(i)
            self.optimize_W1(i)
            self.optimize_W2(i)
            self.contract_bottom_environment(i)
        if self.Ws[2*(self.Ly-1)+1] is None:
            self.optimize_T_W1(self.Ly-1)
        else:
            self.optimize_T(self.Ly-1)
            self.optimize_W1(self.Ly-1)
            self.optimize_W2(self.Ly-1)

    def optimize_column_sweep_top_to_bottom(self):
        """
        Sweeps once from the top to the bottom, optimizing T and W tensors and updating the top environments along the way.
        The upper most tensors (position 2*Ly-1) are not updated. Assumes all thetas have been computed already.
        """
        #print("Column sweep top to bottom")
        if self.Es_top[self.Ly-2] is None:
            self.contract_top_environment(self.Ly-1)
        for i in range(self.Ly-2, 0, -1):
            self.optimize_T(i)
            self.optimize_W2(i)
            self.optimize_W1(i)
            self.contract_top_environment(i)
        if self.Ws[0] is None:
            self.optimize_T_W2(0)
        else:
            self.optimize_T(0)
            self.optimize_W2(0)
            self.optimize_W1(0)

    def optimize_Ts_bottom_to_top(self):
        """
        Sweeps once from the bottom to the top, optimizing T tensors and updating the bottom environments along the way.
        The bottom most tensor (position 0) is not updated. Assumes all thetas have been computed already.
        """
        if self.Es_bot[1] is None:
            self.contract_bottom_environment(0)
        for i in range(1, self.Ly-1):
            self.optimize_T(i)
            self.contract_bottom_environment(i)
        if self.Ws[2*self.Ly-1] is not None:
            self.optimize_T(self.Ly-1)

    def optimize_Ts_top_to_bottom(self):
        """
        Sweeps once from the top to the bottom, optimizing T tensors and updating the top environments along the way.
        The upper most tensor (position 2*Ly-1) is not updated. Assumes all thetas have been computed already.
        """
        if self.Es_top[self.Ly-2] is None:
            self.contract_top_environment(self.Ly-1)
        for i in range(self.Ly-2, 0, -1):
            self.optimize_T(i)
            self.contract_top_environment(i)
        if self.Ws[0] is not None:
            self.optimize_T(0)

    def optimize_Ws_bottom_to_top(self):
        """
        Sweeps once from the bottom to the top, optimizing all W tensors along the way and moving the orthogonality
        center with the sweep. Assumes the orthogonality center is at the bottom at the start.
        """
        #print("Optimizing Ws bottom to top ...")
        #print(f"OC at {self.ortho_center} ...")
        assert self.ortho_center == 0 or self.ortho_center == 1 and self.Ws[0] is None
        if self.ortho_center == 0:
            self.optimize_W1(0)
            self.move_ortho_center_up()
            self.optimize_W2(0)
            self.move_ortho_center_up()
        else:
            self.optimize_T_W2(0)
            self.move_ortho_center_up()
        self.contract_bottom_environment(0)
        for i in range(1, self.Ly-1):
            self.optimize_W1(i)
            self.move_ortho_center_up()
            self.optimize_W2(i)
            self.move_ortho_center_up()
            self.contract_bottom_environment(i)
        if self.Ws[2*self.Ly-1] is not None:
            self.optimize_W1(self.Ly-1)
            self.move_ortho_center_up()
            self.optimize_W2(self.Ly-1)
        else:
            self.optimize_T_W1(self.Ly-1)

    def optimize_Ws_top_to_bottom(self):
        """
        Sweeps once from the top to the bottom, optimizing all W tensors along the way and moving the orthogonality
        center with the sweep. Assumes the orthogonality center is at the top at the start.
        """
        #print("Optimizing Ws top to bottom ...")
        #print(f"OC at {self.ortho_center} ...")
        assert self.ortho_center == 2*self.Ly-1 or self.ortho_center == 2*self.Ly-2 and self.Ws[2*self.Ly-1] is None
        if self.ortho_center == 2*self.Ly-1:
            self.optimize_W2(self.Ly-1)
            self.move_ortho_center_down()
            self.optimize_W1(self.Ly-1)
            self.move_ortho_center_down()
        else:
            self.optimize_T_W1(self.Ly-1)
            self.move_ortho_center_down()
        self.contract_top_environment(self.Ly-1)
        for i in range(self.Ly-2, 0, -1):
            self.optimize_W2(i)
            self.move_ortho_center_down()
            self.optimize_W1(i)
            self.move_ortho_center_down()
            self.contract_top_environment(i)
        if self.Ws[0] is not None:
            #self.optimize_W2(0)
            self.move_ortho_center_down()
            #self.optimize_W1(0)
        else:
            self.optimize_T_W2(0)
            pass

    def optimize_column_sweep_bottom_to_top_moving_oc(self):
        """
        Sweeps once from the bottom to the top, optimizing T and W tensors and updating the bottom environments along the way.
        The bottom most tensors (position 0) are not updated. Assumes all thetas have been computed already and that the OC is at
        the bottom-most position.
        """
        assert self.ortho_center == 0 or self.ortho_center == 1
        self.move_ortho_center_to(2)
        self.contract_bottom_environment(0)
        for i in range(1, self.Ly-1):
            self.optimize_T(i)
            self.optimize_W1(i)
            self.move_ortho_center_up()
            self.optimize_W2(i)
            self.move_ortho_center_up()
            self.contract_bottom_environment(i)
        if self.Ws[2*(self.Ly-1)+1] is None:
            self.optimize_T_W1(self.Ly-1)
        else:
            self.optimize_T(self.Ly-1)
            self.optimize_W1(self.Ly-1)
            self.move_ortho_center_up()
            self.optimize_W2(self.Ly-1)

    def optimize_column_sweep_top_to_bottom_moving_oc(self):
        """
        Sweeps once from the top to the bottom, optimizing T and W tensors and updating the top environments along the way.
        The upper most tensors (position Ly-1) are not updated. Assumes all thetas have been computed already and that the OC is at
        the upper-most position.
        """
        assert self.ortho_center == 2*(self.Ly-1) or self.ortho_center == 2*(self.Ly-1) + 1
        self.move_ortho_center_to(2*self.Ly-3)
        self.contract_top_environment(self.Ly-1)
        for i in range(self.Ly-2, 0, -1):
            self.optimize_T(i)
            self.optimize_W2(i)
            self.move_ortho_center_down()
            self.optimize_W1(i)
            self.move_ortho_center_down()
            self.contract_top_environment(i)
        if self.Ws[0] is None:
            self.optimize_T_W2(0)
        else:
            self.optimize_T(0)
            self.optimize_W2(0)
            self.move_ortho_center_down()
            self.optimize_W2(0)

    def optimize_column_without_moving_oc(self, N_sweeps):
        """
        Variationally optimizes the column, sweeping N_sweeps times up and down.
        """
        # Initial contractions
        if self.thetas[0] is None:
            self.compute_thetas()
        initial_error = None
        errors = []
        # Depending on the position of the orthogonality center, sweep up-down or down-up
        N_Ws = len(self.Ws)
        if self.ortho_center == 0 or self.ortho_center == 1 and self.Ws[0] is None:
            # Start at the bottom, sweep up and down again
            self.compute_top_environments()
            initial_error = self.eps
            if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                errors.append(self.eps)
            if self.debug_logger.log_column_error_yb_before_variational_optimization:
                self.debug_logger.append_to_log_list("column_errors_yb_before_variational_optimization", self.eps)
            self.contract_bottom_environment(0)
            for n in range(N_sweeps):
                # Optimize tensors
                self.optimize_column_sweep_bottom_to_top()
                self.contract_top_environment(self.Ly-1)
                self.optimize_column_sweep_top_to_bottom()
                # Log error
                if self.debug_logger.variational_column_optimization_log_info_per_iteration or self.relative_improvement_stopping_criterion is not None:
                    self.contract_top_environment(0)
                    errors.append(self.eps)
                # Check stopping criterion
                if self.relative_improvement_stopping_criterion is not None and len(errors) > 1 and (errors[-1] <= 1e-14 or np.abs((errors[-1]-errors[-2])/errors[-1]) < self.relative_improvement_stopping_criterion):
                    break
            if self.debug_logger.log_column_error_yb_after_variational_optimization:
                if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", errors[-1])
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", 1.0 if initial_error == 0.0 else errors[-1]/initial_error)
                else:
                    self.contract_top_environment(0)
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", self.eps)
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", 1.0 if initial_error == 0.0 else self.eps/initial_error)
        elif self.ortho_center == N_Ws-1 or self.ortho_center == N_Ws-2 and self.Ws[N_Ws-1] is None:
            # Start at the top, sweep down and up again
            self.compute_bottom_environments()
            initial_error = self.eps
            if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                errors.append(self.eps)
            if self.debug_logger.log_column_error_yb_before_variational_optimization:
                self.debug_logger.append_to_log_list("column_errors_yb_before_variational_optimization", self.eps)
            self.contract_top_environment(self.Ly-1)
            for n in range(N_sweeps):
                # Optimize tensors
                self.optimize_column_sweep_top_to_bottom()
                self.contract_bottom_environment(0)
                self.optimize_column_sweep_bottom_to_top()
                # Log error
                if self.debug_logger.variational_column_optimization_log_info_per_iteration or self.relative_improvement_stopping_criterion is not None:
                    self.contract_bottom_environment(self.Ly-1)
                    errors.append(self.eps)
                # Check stopping criterion
                if self.relative_improvement_stopping_criterion is not None and len(errors) > 1 and (errors[-1] <= 1e-14 or np.abs((errors[-1]-errors[-2])/errors[-1]) < self.relative_improvement_stopping_criterion):
                    break
            if self.debug_logger.log_column_error_yb_after_variational_optimization:
                if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", errors[-1])
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", errors[-1]/initial_error)
                else:
                    self.contract_bottom_environment(self.Ly-1)
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", self.eps)
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", self.eps/initial_error)
        else:
            raise ValueError(f"VariationalColumnOptimizer.optimize_column() must only be called when the orthogonality center is either at the top or at the bottom.")
        if self.debug_logger.variational_column_optimization_log_info:
            self.debug_logger.append_to_log_list(("variational_column_optimization_info", "N_sweeps"), n+1)
        if self.debug_logger.variational_column_optimization_log_info_per_iteration:
            self.debug_logger.append_to_log_list(("variational_column_optimization_info", "column_errors"), errors)

    def optimize_column_move_oc_first_T_then_W(self, N_sweeps):
        """
        Variationally optimizes the column, sweeping N_sweeps times up and down.
        """
        # Initial contractions
        if self.thetas[0] is None:
            self.compute_thetas()
        initial_error = None
        errors = []
        # Depending on the position of the orthogonality center, sweep up-down or down-up
        N_Ws = len(self.Ws)
        if self.ortho_center == 0 or self.ortho_center == 1 and self.Ws[0] is None:
            #print("Start at the bottom, sweep up and down again")
            # Start at the bottom, sweep up and down again
            self.compute_top_environments()
            initial_error = self.eps
            if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                errors.append(self.eps)
            if self.debug_logger.log_column_error_yb_before_variational_optimization:
                self.debug_logger.append_to_log_list("column_errors_yb_before_variational_optimization", self.eps)
            self.contract_bottom_environment(0)            
            for n in range(N_sweeps):
                # Optimize T tensors
                self.optimize_Ts_bottom_to_top()
                self.contract_top_environment(self.Ly-1)
                self.optimize_Ts_top_to_bottom()
                self.contract_bottom_environment(0)
                # Optimize W tensors
                self.optimize_Ws_bottom_to_top()
                self.optimize_Ws_top_to_bottom()
                # Log error
                if self.debug_logger.variational_column_optimization_log_info_per_iteration or self.relative_improvement_stopping_criterion is not None:
                    self.contract_top_environment(0)
                    errors.append(self.eps)
                # Check stopping criterion
                if self.relative_improvement_stopping_criterion is not None and len(errors) > 1 and (errors[-1] <= 1e-14 or np.abs((errors[-1]-errors[-2])/errors[-1]) < self.relative_improvement_stopping_criterion):
                    break
            if self.debug_logger.log_column_error_yb_after_variational_optimization:
                if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", errors[-1])
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", 1.0 if initial_error == 0.0 else errors[-1]/initial_error)
                else:
                    self.contract_top_environment(0)
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", self.eps)
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", 1.0 if initial_error == 0.0 else self.eps/initial_error)
        elif self.ortho_center == N_Ws-1 or self.ortho_center == N_Ws-2 and self.Ws[N_Ws-1] is None:
            #print("Start at the top, sweep down and up again")
            # Start at the top, sweep down and up again
            self.compute_bottom_environments()
            initial_error = self.eps
            if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                errors.append(self.eps)
            if self.debug_logger.log_column_error_yb_before_variational_optimization:
                self.debug_logger.append_to_log_list("column_errors_yb_before_variational_optimization", self.eps)
            self.contract_top_environment(self.Ly-1)
            for n in range(N_sweeps):
                # Optimize T tensors
                self.optimize_Ts_top_to_bottom()
                self.contract_bottom_environment(0)
                self.optimize_Ts_bottom_to_top()
                self.contract_top_environment(self.Ly-1)
                # Optimize W tensors
                self.optimize_Ws_top_to_bottom()
                self.optimize_Ws_bottom_to_top()
                # Log error
                if self.debug_logger.variational_column_optimization_log_info_per_iteration or self.relative_improvement_stopping_criterion is not None:
                    self.contract_top_environment(self.Ly-1)
                    errors.append(self.eps)
                # Check stopping criterion
                if self.relative_improvement_stopping_criterion is not None and len(errors) > 1 and (errors[-1] <= 1e-14 or np.abs((errors[-1]-errors[-2])/errors[-1]) < self.relative_improvement_stopping_criterion):
                    break
            if self.debug_logger.log_column_error_yb_after_variational_optimization:
                if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", errors[-1])
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", errors[-1]/initial_error)
                else:
                    self.contract_bottom_environment(self.Ly-1)
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", self.eps)
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", self.eps/initial_error)
        else:
            raise ValueError(f"VariationalColumnOptimizer.optimize_column() must only be called when the orthogonality center is either at the top or at the bottom.")
        if self.debug_logger.variational_column_optimization_log_info:
            self.debug_logger.append_to_log_list(("variational_column_optimization_info", "N_sweeps"), n+1)
        if self.debug_logger.variational_column_optimization_log_info_per_iteration:
            self.debug_logger.append_to_log_list(("variational_column_optimization_info", "column_errors"), errors)

    def optimize_column_move_oc_T_and_Ws_alternatingly(self, N_sweeps):
        """
        Variationally optimizes the column, sweeping N_sweeps times up and down.
        """
        # Initial contractions
        if self.thetas[0] is None:
            self.compute_thetas()
        initial_error = None
        errors = []
        # Depending on the position of the orthogonality center, sweep up-down or down-up
        N_Ws = len(self.Ws)
        if self.ortho_center == 0 or self.ortho_center == 1 and self.Ws[0] is None:
            #print("Start at the bottom, sweep up and down again")
            # Start at the bottom, sweep up and down again
            self.compute_top_environments()
            initial_error = self.eps
            if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                errors.append(self.eps)
            if self.debug_logger.log_column_error_yb_before_variational_optimization:
                self.debug_logger.append_to_log_list("column_errors_yb_before_variational_optimization", self.eps)
            self.contract_bottom_environment(0)            
            for n in range(N_sweeps):
                # Optimize tensors
                self.optimize_column_sweep_bottom_to_top_moving_oc()
                self.optimize_column_sweep_top_to_bottom_moving_oc()
                # Log error
                if self.debug_logger.variational_column_optimization_log_info_per_iteration or self.relative_improvement_stopping_criterion is not None:
                    self.contract_top_environment(0)
                    errors.append(self.eps)
                # Check stopping criterion
                if self.relative_improvement_stopping_criterion is not None and len(errors) > 1 and (errors[-1] <= 1e-14 or np.abs((errors[-1]-errors[-2])/errors[-1]) < self.relative_improvement_stopping_criterion):
                    break
            if self.debug_logger.log_column_error_yb_after_variational_optimization:
                if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", errors[-1])
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", 1.0 if initial_error == 0.0 else errors[-1]/initial_error)
                else:
                    self.contract_top_environment(0)
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", self.eps)
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", 1.0 if initial_error == 0.0 else self.eps/initial_error)
        elif self.ortho_center == N_Ws-1 or self.ortho_center == N_Ws-2 and self.Ws[N_Ws-1] is None:
            #print("Start at the top, sweep down and up again")
            # Start at the top, sweep down and up again
            self.compute_bottom_environments()
            initial_error = self.eps
            if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                errors.append(self.eps)
            if self.debug_logger.log_column_error_yb_before_variational_optimization:
                self.debug_logger.append_to_log_list("column_errors_yb_before_variational_optimization", self.eps)
            self.contract_top_environment(self.Ly-1)
            for n in range(N_sweeps):
                # Optimize tensors
                self.optimize_column_sweep_top_to_bottom_moving_oc()
                self.optimize_column_sweep_bottom_to_top_moving_oc()
                self.optimize_column_sweep_bottom_to_top()
                # Log error
                if self.debug_logger.variational_column_optimization_log_info_per_iteration or self.relative_improvement_stopping_criterion is not None:
                    self.contract_top_environment(self.Ly-1)
                    errors.append(self.eps)
                # Check stopping criterion
                if self.relative_improvement_stopping_criterion is not None and len(errors) > 1 and (errors[-1] <= 1e-14 or np.abs((errors[-1]-errors[-2])/errors[-1]) < self.relative_improvement_stopping_criterion):
                    break
            if self.debug_logger.log_column_error_yb_after_variational_optimization:
                if self.debug_logger.variational_column_optimization_log_info_per_iteration:
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", errors[-1])
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", errors[-1]/initial_error)
                else:
                    self.contract_bottom_environment(self.Ly-1)
                    self.debug_logger.append_to_log_list("column_errors_yb_after_variational_optimization", self.eps)
                    self.debug_logger.append_to_log_list("column_errors_yb_improvement_through_variational_optimization", self.eps/initial_error)
        else:
            raise ValueError(f"VariationalColumnOptimizer.optimize_column() must only be called when the orthogonality center is either at the top or at the bottom.")
        if self.debug_logger.variational_column_optimization_log_info:
            self.debug_logger.append_to_log_list(("variational_column_optimization_info", "N_sweeps"), n+1)
        if self.debug_logger.variational_column_optimization_log_info_per_iteration:
            self.debug_logger.append_to_log_list(("variational_column_optimization_info", "column_errors"), errors)

    def optimize_column(self, N_sweeps):
        if self.mode == "dont_move_oc":
            self.optimize_column_without_moving_oc(N_sweeps)
        elif self.mode == "move_oc_first_T_then_W":
            self.optimize_column_move_oc_first_T_then_W(N_sweeps)
        elif self.mode == "move_oc_alternate_T_W":
            self.optimize_column_move_oc_T_and_Ws_alternatingly(N_sweeps)
        else:
            raise ValueError(f"\"{self.mode}\" is not a valid mode for calling VariationalColumnOptimizer.optimize_column()!")