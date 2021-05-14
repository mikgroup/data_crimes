'''
This module includes functions for MRI reconstruction using Dictionary Learning.

It is based on Jon Tamir's tutorial:
"MRI Reconstruction using Learned Dictionaries" ISMRM 2020 Educational Session.
and associated code:
https://github.com/utcsilab/dictionary_learning_ismrm_2020

(c) Jon Tamir, 2020

'''

import numpy as np
import sigpy as sp
from sigpy.mri.linop import Sense


def omp_single(yi, D, num_nonzero_coeffs=3, use_sigpy=False, device=None):
    """Orthogonal Matching Pursuit (OMP) for a data point.
    Args:
        yi: [N, 1] data point
        D: [N, P] dictionary
        num_nonzero_coeffs: K in notation above
        device: SigPy device (cpu or gpu)
    Notes:
        Not fully tested on GPU
        Not vectorized for speed
    """

    device = sp.get_device(yi)
    xp = sp.Device(device).xp

    # initialize residual to y
    resid = yi
    idxs = []
    for k in range(num_nonzero_coeffs):
        # find dictionary most correlated with residual
        z = abs(xp.conj(D.T).dot(resid))
        ii = int(xp.argmax(z))
        idxs = idxs + [ii]
        Dsub = D[:, idxs]
        # solve for coefficients
        if use_sigpy:
            A_D = sp.linop.MatMul([len(idxs), 1], Lsub)
            vals = sp.app.LinearLeastSquares(A_D, yi, show_pbar=False).run().ravel()
        else:
            if device is None or device == sp.cpu_device:
                rcond = None
            else:
                rcond = -1
            vals = xp.linalg.lstsq(Dsub, yi, rcond=rcond)[0]
        # update residual
        resid = yi - Dsub.dot(vals)
    return idxs, vals.ravel()


class OMP(sp.alg.Alg):
    """Orthogonal Matching Pursuit (OMP) algorithm for a batch of data.
    Each iteration of the algorithm processes the next data point, for
    a total of L iterations
    Args:
        y: [N, L] data
        D: [N, P] dictionary
        num_nonzero_coeffs: K in notation above
        DC: whether to include a DC dictionary element
    """

    def __init__(
            self,
            y,
            D,
            num_nonzero_coeffs,
            device=sp.cpu_device,
            DC=False,
            **kwargs):
        self.max_iter = y.shape[1]

        self.y = sp.to_device(y, device)
        self.D = sp.to_device(D, device)
        self.num_filters = self.D.shape[-1]
        self.num_nonzero_coeffs = num_nonzero_coeffs
        self.device = sp.Device(device)
        self.DC = DC

        self.dtype = self.y.dtype
        self.num_points, self.num_data = self.y.shape
        self.A_shape = (self.num_filters, self.num_data)

        xp = self.device.xp
        self.A = xp.zeros((self.num_filters, self.num_data), dtype=self.y.dtype)
        self.iter = 0

        super().__init__(self.max_iter)

    def _update(self):
        xp = self.device.xp
        yi = self.y[:, self.iter]
        idxs, vals = omp_single(yi, self.D, self.num_nonzero_coeffs, device=self.device)
        self.A[idxs, self.iter] = vals.ravel()

    def _done(self):
        return (self.iter >= self.max_iter)


####################################################################################

class SparseDecom(sp.app.App):
    """Compute the sparse coefficients for a dataset given a fixed dictionary
    Args:
        y: [N, L] data
        D: [N, P] dictionary
        lamda: sparsity level (if mode='omp') or l1-regularization parameter (if mode='l1')
        mode: 'l1' for l1-regularization or 'omp' for OMP
        DC: whether to include a DC dictionary element
        device: CPU or GPU
    """
    def __init__(
            self,
            y,
            D,
            lamda=0.005,
            mode='l1',
            DC=False,
            device=sp.cpu_device,
            show_pbar=True,
            leave_pbar=True,
            **kwargs):

        self.y = sp.to_device(y, device)
        self.D = sp.to_device(D, device)
        self.lamda = lamda
        self.device = device
        self.mode = mode
        self.show_pbar = show_pbar
        self.leave_pbar = leave_pbar

        self._get_params()

        if self.mode is 'omp':
            self.alg = OMP(self.y,
                           self.D,
                           self.lamda,
                           DC=DC,
                           device=self.device)
        elif self.mode is 'l1':
            self.A = sp.linop.MatMul(
                self.A_shape,
                self.D)

            proxg_A = sp.prox.L1Reg(self.A_shape, lamda)

            self.alg = sp.app.LinearLeastSquares(self.A,
                                                 self.y,
                                                 proxg=proxg_A,
                                                 show_pbar=self.show_pbar,
                                                 **kwargs).alg


        super().__init__(self.alg, show_pbar=self.show_pbar, leave_pbar=self.leave_pbar)

        self._output = self.__output


    def _get_params(self):
        self.device = sp.Device(self.device)
        self.dtype = self.y.dtype
        self.num_data = self.y.shape[-1]
        self.num_filters = self.D.shape[-1]
        self.num_points = self.D.shape[0]

        self.A_shape = (self.num_filters, self.num_data)

    def __output(self):
        if self.mode == 'omp':
            return self.alg.A
        else:
            return self.alg.x


def nrmse(x, y):
    """Calculate normalized root mean-squared error (for complex data):
    || x - y||_2 / ||x||_2
    """
    xp = sp.get_device(x).xp
    return xp.linalg.norm(x.ravel() - y.ravel()) / xp.linalg.norm(x.ravel())


####################################################################################

class DictionaryLearning(sp.app.App):
    """Dictionary Learning. Solves the optimization problem \min_{A, D} || X - AD ||_2^2 + R1(A) + R2(D),
    where R2(D) enforces l2-norm of each column of D, and R1(A) enforces sparsity of A.
    R1(A) supports either OMP or L1-minimization.
    To update D, supports either linear least-squares, K-SVD, or approximate K-SVD

    Args:
        y: [N, L] data
        num_filters: number of dictionary atoms
        batch_size: batch size for each alternating minimization outer loop
        lamda: sparsity level (if A_mode='omp') or l1-regularization parameter (if A_mode='l1')
        alpha: l2-regularization for dictionary update (if L_mode='lls')
        A_mode: How to update sparse codes.
            'l1' for l1-regulaization or 'omp' for OMP
        D_mode: How to update dictionary.
            'lls' for linear least squares,'ksvd' for K-SVD, or 'aksvd' for approximate K-SVD
        D_init_mode: How to initialize the dictionary.
            'random' for random, 'svd' for singular value decomposition, 'data' for using the data itself
        DC: whether to include a DC dictionary element
        skip_final_decomp: True to not run the final SparseDecomposition on the full data
        max_inner_iter: How many inner iterations (for A_mode='l1' and D_mode='lls')
        max_power_iter: How many power iterations (for A_mode='l1' and D_mode='lls')
        max_iter: How many alternating minimization outer iterations
        device: CPU or GPU
    """

    def __init__(
            self,
            y,
            num_filters,
            batch_size,
            lamda=0.001,
            alpha=0.5,
            A_mode='omp',
            D_mode='ksvd',
            mask_idx=None,
            D_init_mode='random',
            DC=False,
            skip_final_decomp=False,
            max_inner_iter=100,
            max_power_iter=10,
            max_iter=10,
            device=sp.cpu_device,
            **kwargs):
        self.y = sp.to_device(y, device)
        self.num_filters = num_filters
        self.batch_size = batch_size
        self.lamda = lamda
        self.alpha = alpha
        self.A_mode = A_mode
        self.D_mode = D_mode
        self.max_inner_iter = max_inner_iter
        self.max_power_iter = max_power_iter
        self.max_iter = max_iter
        self.device = device
        self.D_init_mode = D_init_mode
        self.DC = DC
        self.skip_final_decomp = skip_final_decomp
        self.mask_idx = mask_idx
        self.current_resid = []

        self._get_params()
        self._get_vars()
        self._get_alg()

        super().__init__(self.alg, show_pbar=True, **kwargs)

    def _get_params(self):
        self.device = sp.Device(self.device)
        self.dtype = self.y.dtype
        self.num_points, self.num_data = self.y.shape
        self.batch_size = min(self.num_data, self.batch_size)
        self.num_batches = self.num_data // self.batch_size
        self.t_start = 0
        self.batch_ridx = np.random.permutation(self.num_data)

        self.D_shape = (self.num_points, self.num_filters)
        self.A_t_shape = (self.num_filters, self.batch_size)

    def _get_vars(self):
        xp = self.device.xp
        with self.device:
            # storage for batch size amount of data
            self.y_t = xp.empty((self.num_points, self.batch_size),
                                dtype=self.dtype)

            # Dictionary initialization
            if self.D_init_mode == 'random':
                self.D = sp.randn(self.D_shape, dtype=self.dtype,
                                  device=self.device)
            elif self.D_init_mode == 'svd':
                uu, ss, vv = xp.linalg.svd(self.y, full_matrices=False)
                self.D = vv[:, :self.num_filters]
            elif self.D_init_mode == 'data':
                ridx = np.random.permutation(self.num_data)
                self.D = self.y[:, ridx[:self.num_filters]]
            else:
                self.D = sp.to_device(self.D_init_mode, self.device)

            # Normalize each dictionary atom
            if self.DC:
                self.D -= xp.mean(self.D, axis=0, keepdims=True)
                self.D[:, 0] = 1

            self.D /= xp.sum(xp.abs(self.D) ** 2,
                             axis=(0),
                             keepdims=True) ** 0.5

            self.D_old = xp.empty(self.D_shape, dtype=self.dtype)

            self.A = xp.zeros((self.num_filters, self.num_data), dtype=self.dtype)
            self.A_t = xp.zeros((self.num_filters, self.batch_size), dtype=self.dtype)

    def _get_alg(self):

        def min_A_t():
            self.A_t = SparseDecom(
                self.y_t,
                self.D,
                lamda=self.lamda,
                mode=self.A_mode,
                DC=self.DC,
                max_power_iter=self.max_power_iter,
                max_iter=self.max_inner_iter,
                device=self.device,
                show_pbar=False).run()

        def min_D_lls():
            assert not self.DC, 'not implemented'

            self.Aop_D = sp.linop.RightMatMul(
                self.D_shape,
                self.A_t)

            mu = (1 - self.alpha) / self.alpha

            proxg_D = sp.prox.L2Proj(
                self.D.shape, 1., axes=[0])

            sp.app.LinearLeastSquares(self.Aop_D, self.y_t, x=self.D,
                                      z=self.D_old,
                                      proxg=proxg_D,
                                      lamda=mu,
                                      max_power_iter=self.max_power_iter,
                                      max_iter=self.max_inner_iter, show_pbar=False).run()

        def min_D_ksvd():
            if self.D_mode == 'ksvd':
                approx = False
            else:
                approx = True
            for k in np.random.permutation(self.num_filters):
                if k > 0 or (not self.DC):
                    self.ksvd(k, approx=approx)

        if self.D_mode == 'lls':
            min_D = min_D_lls
        else:
            min_D = min_D_ksvd

        self.alg = sp.alg.AltMin(min_A_t, min_D, max_iter=self.max_iter)

    def ksvd(self, k, approx=False):
        r""" K-SVD algorithm.
        Supports either full K-SVD, or approximate K-SVD.
        """
        xp = self.device.xp
        lk = self.D[:, k][:, None]
        ak = self.A_t[k, :][None, :]
        idx = ak.ravel() != 0
        if np.sum(idx) > 0:
            if approx:
                self.D[:, k] = 0
                g = ak[:, idx].T
                yI = self.y_t[:, idx]
                AI = self.A_t[:, idx]
                d = yI.dot(g) - self.D.dot(AI).dot(g)
                d /= xp.linalg.norm(d)
                g = yI.T.dot(d) - xp.conj(self.D.dot(AI)).T.dot(d)
                self.D[:, k] = d.ravel()
                self.A_t[k, idx] = g.T
            else:
                Ek = self.y_t - self.D.dot(self.A_t) + lk.dot(ak)
                EkR = Ek[:, idx]
                uu, ss, vv = xp.linalg.svd(EkR, full_matrices=False)
                self.D[:, k] = uu[:, 0]
                self.A_t[k, idx] = vv[0, :] * ss[0]

    def _pre_update(self):

        ridx_t = self.batch_ridx[self.t_start:self.t_start + self.batch_size]

        self.t_start = self.t_start + self.batch_size

        if self.t_start >= self.num_data:
            #             print('reset ridx. t_start is', self.t_start, 'ridx_t is', ridx_t.shape)
            self.batch_ridx = np.random.permutation(self.num_data)

            if len(ridx_t) < self.batch_size:
                self.t_start = self.batch_size - len(ridx_t)
                ridx_t = np.concatenate((ridx_t, self.batch_ridx[:self.t_start]))
            else:
                self.t_start = 0

        sp.copyto(self.y_t, self.y[:, ridx_t])
        sp.copyto(self.D_old, self.D)

    def _summarize(self):
        if self.show_pbar:
            xp = self.device.xp
            self.current_resid.append(sp.to_device(xp.linalg.norm(self.y_t - self.D.dot(self.A_t)), sp.cpu_device))
            self.pbar.set_postfix(resid='{0:.2E}'.format(self.current_resid[-1]))

    def _output(self):
        if not self.skip_final_decomp:
            self.A = SparseDecom(
                self.y,
                self.D,
                lamda=self.lamda,
                mode=self.A_mode,
                max_power_iter=self.max_power_iter,
                max_iter=self.max_inner_iter,
                device=self.device,
                show_pbar=True,
                leave_pbar=False).run()
        return self.D, self.A

####################################################################

class DictionaryLearningMRI(sp.app.App):
    """Dictionary Learning. Solves the optimization problem \min_{A, D} || X - AD ||_2^2 + R1(A) + R2(D),
    where R2(D) enforces l2-norm of each column of D, and R1(A) enforces sparsity of A.
    R1(A) supports either OMP or L1-minimization.
    To update D, supports either linear least-squares, K-SVD, or approximate K-SVD

    Args:
        ksp: [nc, nx, ny] of [nx, ny] kspace with optional multi-channel dimension
        mask: [nx, ny] Cartesian sampling pattern
        maps: [nc, nx, ny] sensitivity maps if multi-channel, None otherwise
        num_filters: number of dictionary atoms
        batch_size: batch size for each alternating minimization outer loop
        block_shape: image patch shape
        block_strides: image patch strides
        lamda: sparsity level (if A_mode='omp') or l1-regularization parameter (if A_mode='l1')
        alpha: l2-regularization for dictionary update (if L_mode='lls')
        nu: regularization parameter trading off data consistency vs. approximation error
        A_mode: How to update sparse codes.
            'l1' for l1-regulaization or 'omp' for OMP
        D_mode: How to update dictionary.
            'lls' for linear least squares,'ksvd' for K-SVD, or 'aksvd' for approximate K-SVD
        D_init_mode: How to initialize the dictionary.
            'random' for random, 'svd' for singular value decomposition, 'data' for using the data itself
        DC: whether to include a DC dictionary element
        max_inner_iter: How many inner iterations (for A_mode='l1' and D_mode='lls')
        max_power_iter: How many power iterations (for A_mode='l1' and D_mode='lls')
        max_iter: How many alternating minimization outer iterations
        img_ref: optional ground-truth image to calculate NRMSE each iteration
        device: CPU or GPU
    """

    def __init__(
            self,
            ksp,  # [nc, nx, ny]
            mask,  # [nx, ny]
            maps,
            num_filters,
            batch_size,
            block_shape,  # [bx, by]
            block_strides,  # [sx sy]
            lamda=0.001,
            alpha=0.5,
            nu=.01,
            A_mode='omp',
            D_mode='ksvd',
            D_init_mode='random',
            DC=False,
            max_inner_iter=100,
            max_power_iter=10,
            max_iter=10,
            img_ref=None,
            device=sp.cpu_device,
            **kwargs):

        self.ksp = sp.to_device(ksp, device)
        self.mask = sp.to_device(mask, device)

        self.maps = maps
        if self.maps != None:
            self.maps = sp.to_device(maps, device)

        if img_ref is None:
            self.img_ref = None
        else:
            self.img_ref = sp.to_device(img_ref, device)

        self.num_filters = num_filters
        self.batch_size = batch_size
        self.block_shape = block_shape
        self.block_strides = block_strides
        self.nu = nu
        self.lamda = lamda
        self.alpha = alpha
        self.A_mode = A_mode
        self.D_mode = D_mode
        self.max_inner_iter = max_inner_iter
        self.max_power_iter = max_power_iter
        self.max_iter = max_iter
        self.device = device
        self.D_init_mode = D_init_mode
        self.DC = DC

        self.first = True
        self.residuals = []
        self.nrmse_vals = []
        self.SSIM_vals = []

        self._get_params()
        self._get_vars()
        self._get_alg()

        super().__init__(self.alg, **kwargs)

    def _get_params(self):
        self.device = sp.Device(self.device)
        self.dtype = self.ksp.dtype

        self.ndims = len(self.ksp.shape)
        if self.ndims > 2:
            self.img_shape = self.ksp.shape[-(self.ndims - 1):]
        else:
            self.img_shape = self.ksp.shape

        if self.maps != None:
            fft_sense_op = Sense(self.maps, ishape=self.img_shape)
        else:
            fft_sense_op = sp.linop.FFT(self.ksp.shape, axes=(-1, -2))

        mask_op = sp.linop.Multiply(self.ksp.shape, self.mask)
        self.mri_op = mask_op * fft_sense_op

        block_op = sp.linop.BlocksToArray(self.img_shape, self.block_shape, self.block_strides)

        self.img_blocks_shape = block_op.ishape

        self.num_data = np.prod(self.img_blocks_shape[:2])
        self.num_points = np.prod(self.img_blocks_shape[2:])

        print('num_data:', self.num_data)
        print('num_points:', self.num_points)

        reshape_op = sp.linop.Reshape(block_op.ishape, (self.num_data, self.num_points))

        self.reshape_block_op = block_op * reshape_op

        self.L_shape = (self.num_points, self.num_filters)
        self.R_shape = (self.num_filters, self.num_data)
        #         print(self.L_shape, self.R_shape)

        self.forward_op = self.mri_op * self.reshape_block_op

    #         print(self.forward_op.oshape, self.forward_op.ishape)

    def _get_vars(self):
        xp = self.device.xp

        with self.device:
            self.img_adjoint = self.mri_op.H * self.ksp
            self.img_blocks_flat = (self.forward_op.H * self.ksp).T

            # print(self.img_blocks_flat.shape)

            self.A = xp.zeros((self.num_filters, self.num_data), dtype=self.dtype)
            self.D = xp.zeros((self.num_points, self.num_filters), dtype=self.dtype)
            self.block_scale_factor = self.reshape_block_op * self.reshape_block_op.H * xp.ones(self.img_shape,
                                                                                                dtype=self.dtype)

            if self.img_ref is not None:
                self.img_ref = self.reshape_block_op * self.reshape_block_op.H * self.img_ref
                self.img_ref /= self.block_scale_factor

            img_mask = xp.array(abs(self.img_adjoint) > .1 * max(abs(self.img_adjoint.ravel())), dtype=self.dtype)

            img_mask_blocks = self.reshape_block_op.H * img_mask
            #             print('shape:', img_mask_blocks.shape)
            count = xp.sum(img_mask_blocks, axis=-1)
            self.block_idx = count > 1

    # print(self.img_blocks_flat.shape, self.block_idx.shape)

    def _get_alg(self):

        def min_ksvd():
            if self.first:
                D_init_mode = self.D_init_mode
                self.first = False
            else:
                D_init_mode = self.D
            self.D, _A = DictionaryLearning(self.img_blocks_flat[:, self.block_idx],
                                            self.num_filters,
                                            batch_size=self.batch_size,
                                            A_mode=self.A_mode,
                                            D_mode=self.D_mode,
                                            D_init_mode=D_init_mode,
                                            DC=self.DC,
                                            lamda=self.lamda,
                                            mask_idx=self.block_idx,
                                            alpha=0.5,
                                            max_inner_iter=2000,
                                            max_iter=self.max_inner_iter,
                                            max_power_iter=30,
                                            skip_final_decomp=False,
                                            leave_pbar=False,
                                            device=self.device).run()
            self.A[:, self.block_idx] = _A

        def min_mri():
            img = self.D.dot(self.A)
            img = self.reshape_block_op * img.T
            img /= self.block_scale_factor

            app = sp.app.LinearLeastSquares(self.mri_op,
                                            self.ksp,
                                            lamda=self.nu,
                                            z=img,
                                            max_iter=25,
                                            save_objective_values=True,
                                            show_pbar=True,
                                            leave_pbar=False)
            self.img_out = app.run()
            self.objective_values = app.objective_values
            self.img_blocks_flat = (self.reshape_block_op.H * self.img_out).T

        self.alg = sp.alg.AltMin(min_ksvd, min_mri, max_iter=self.max_iter)

     def _summarize(self):
        if self.show_pbar:
            xp = self.device.xp
            self.residuals.append(self.objective_values[-1])

            # if self.img_ref is None:
            #     val = self.residuals[-1]
            #     self.pbar.set_postfix(resid='{0:.2E}'.format(val))
            # else:
            #     # calc nrmse
            #     val = nrmse(self.img_ref, self.img_out)
            #     self.nrmse_vals.append(val)
            #
            #
            #     # calc SSIM
            #     err = error_metrics(np.abs(self.img_ref), np.abs(self.img_out))
            #     err.calc_SSIM()
            #     self.SSIM_vals.append(err.SSIM)
            #
            #     print('nrmse = {} SSIM={}'.format(val,err.SSIM))
            #
            #     # dispaly wait bar + nrmse + SSIM
            #     #self.pbar.set_postfix(nrmse='{0:.2E} SSIM={0:.2E}'.format(val, err.SSIM))

    def _output(self):
        return self.D, self.A, self.img_out

