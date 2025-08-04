
import cuquantum.densitymat as cudense
import cupy

import qutip
from qutip.core.options import QutipOptions
from qutip.core import data as _data
from qutip.settings import settings
from qutip.core.numpy_backend import np

from .cudense import CuOperator
from .state import CuState
from .ode import *
from .qobjevo import CuQobjEvo
from ..dense import CuPyDense


_data.to.register_group(
    ['cuDensity'], dense=_data.Dense, sparse=CuOperator, diagonal=CuOperator
)


class cuDensityOption(QutipOptions):
    _options = {"ctx": None}
    _settings_name = "cuDensity"
    _properties = {}


cuDensityOption_instance = cuDensityOption()
cuDensityOption_instance._set_as_global_default()


class Result(qutip.Result):
    def _e_op_func(self, e_op):
        if isinstance(e_op, (qutip.Qobj, qutip.QobjEvo)):
            gpu_caller = CuQobjEvo(qutip.QobjEvo(e_op))
            return gpu_caller.expect
        raise NotImplementedError


def set_as_default(ctx):
    settings.cuDensity["ctx"] = ctx
    settings.core["default_dtype"] = "cuDensity"
    settings.core['numpy_backend'] = cupy

    # settings.core["default_dtype_scope"] = "full"
    if True:  # if mpi:
        qutip.settings.core["auto_real_casting"] = False

    qutip.SESolver.solver_options['method'] = "CuVern7"
    qutip.MESolver.solver_options['method'] = "CuVern7"
    qutip.MCSolver.solver_options['method'] = "CuVern7"

    qutip.SESolver._resultclass = Result
    qutip.MESolver._resultclass = Result
    qutip.MCSolver._trajectory_resultclass = Result
