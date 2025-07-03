


import qutip
from qutip.core.options import QutipOptions
from qutip.core import data as _data
from qutip.settings import settings

from .cudense import cuDensity
from .state import cuState


to.register_group(
    ['cuDensity'], dense=cuState, sparse=cuDensity, diagonal=cuDensity
)


class cuDensityOption(QutipOptions):
    _options = {
        "ctx": None,
    }
    _settings_name = "cuDensity"
    _properties = {}


cuDensityOption._set_as_global_default()


def set_as_default(ctx):
    settings.cuDensity["ctx"] = ctx
    settings.core["default_dtype"] = "cudensity"
    settings.core["default_dtype_scope"] = "full"
    qutip.SESolver.solver_options['method'] = "CuVern7"
    qutip.MESolver.solver_options['method'] = "CuVern7"
    qutip.MCSolver.solver_options['method'] = "CuVern7"
