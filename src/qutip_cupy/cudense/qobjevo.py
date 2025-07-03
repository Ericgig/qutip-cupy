

















class CuQobjEvo:
    """
    Pytree friendly QobjEvo for the Diffrax integrator.

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

        for part in as_list:
            if isinstance(part, Qobj):
                self.operator.append(part.data_as("OperatorTerm"))
            elif (
                isinstance(part, list) and isinstance(part[0], Qobj)
            ):
                self.operator.append(part[0].data_as("OperatorTerm"), part[1])
            else:
                raise NotImplementedError(
                    "Function based QobjEvo are not supported"
                )

    def __call__(self, t, _args=None, **kwargs):
        raise NotImplementedError

    def data(self, t):
        raise NotImplementedError

    def matmul_data(self, t, y, out=None):
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
