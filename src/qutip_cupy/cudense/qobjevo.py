try:
    import cuquantum.densitymat as cudense
    Operator = cudense.Operator

except ImportError:
    class _Missing:
        ...

    cudense = None
    Operator = _Missing


from .cuState import zeros_like_cuState


# TODO: Being child class of QobjEvo needed? Or duck typing good enough?
class CuQobjEvo(QobjEvo):
    """
    A QobjEvo using cuDensity's Operator instead of qutip's cython operations.
    Designed to use in solver only, as a static object

    It only support list based `QobjEvo`.
    """

    terms: list
    shape: tuple
    dims: object
    action_ready: Bool
    expect_ready: Bool

    def __init__(self, qobjevo):
        as_list = qobjevo.to_list()
        self.dims = qobjevo.dims
        self.shape = qobjevo.shape
        self.action_ready = False
        self.expect_ready = False
        self.hilbert_space_dims = tuple(self.dims[0][0])

        self.operator = Operator(self.hilbert_space_dims)
        dual = qobjevo._dims.is_super

        for part in as_list:
            if isinstance(part, Qobj):
                self.operator.append(part.data.to_OperatorTerm(
                    dual, hilbert_dims=self.hilbert_space_dims
                ))
            elif (
                isinstance(part, list) and isinstance(part[0], Qobj)
            ):
                qobj = part[0]
                coeff = lambda t, _: part[1](t)
                self.operator.append(qobj.data.to_OperatorTerm(
                    dual, hilbert_dims=self.hilbert_space_dims
                ), coeff)
            else:
                raise NotImplementedError(
                    "Function based QobjEvo are not supported"
                )

    def __call__(self, t, _args=None, **kwargs):
        raise NotImplementedError

    def data(self, t):
        raise NotImplementedError

    def matmul_data(self, t, y, out=None):
        if not isinstance(y, CuState):
            y = CuState(y, hilbert_dims=elf.hilbert_space_dims)
        if not self.action_ready:
            self.operator.prepare_action(
                settings.cuDensity["ctx"],
                y.base
            )
            self.action_ready = True
        if out is None:
            out = zeros_like_cuState(y)
        self.operator.compute_action(
            t,
            state=y.base,
            state_out=out.base,
        )
        return out

    def expect_data(self, t, y, out=None):
        if not isinstance(y, CuState):
            y = CuState(y, hilbert_dims=elf.hilbert_space_dims)
        if not self.expect_ready:
            self.operator.prepare_expectation(
                settings.cuDensity["ctx"],
                y.base
            )
            self.expect_ready = True
        if out is None:
            out = zeros_like_cuState(y)
        self.operator.compute_expectation(t, state=y.base)
        return out

    def arguments(self, args):
        raise NotImplementedError

    def linear_map(self, op_mapping, *, _skip_check=False):
        raise NotImplementedError

    def tidyup(self, atol=1e-12):
        raise NotImplementedError

    def to(self, data_type):
        raise NotImplementedError

    def dag(self):
        raise NotImplementedError

    def conj(self):
        raise NotImplementedError

    def trans(self):
        raise NotImplementedError

    @property
    def dtype(self):
        return Operator

    @property
    def num_elements(self):
        raise NotImplementedError

    @property
    def isconstant(self):
        raise NotImplementedError
