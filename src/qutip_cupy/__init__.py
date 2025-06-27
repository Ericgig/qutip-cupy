""" The qutip-cupy package provides a CuPy-based data layer for QuTiP. """

# we need to silence this specific warning
# remember to remove once QuTiP moves matplotlib
# to an official optional dependency
import warnings

try:
    __import__("cupy")
except ModuleNotFoundError:
    raise RuntimeError(
        "qutip_cupy requires cupy to be installed, please install cupy by following "
        "the instructions at https://docs.cupy.dev/en/stable/install.html"
    )

with warnings.catch_warnings():
    warnings.filterwarnings(
        action="ignore", category=UserWarning, message=r"matplotlib not found:"
    )

    from qutip.core import data

# qutip_cupy imports need to be after the cupy import check above
from .version import version as __version__  # noqa: E402
from . import dense as cd  # noqa: E402
from . import dia as cdia  # noqa: E402
from . import dense_functions as cdf  # noqa: E402
from . import dia_functions as cdiaf  # noqa: E402
from . import linalg  # noqa: E402
from .cudense import CuOperator

__all__ = ["__version__", "CuPyDense", "CuPyDia", "CuOperator"]

CuPyDense = cd.CuPyDense
CuPyDia = cdia.CuPyDia

data.to.add_conversions(
    [
        (CuPyDense, data.Dense, cd.cupydense_from_dense),
        (data.Dense, CuPyDense, cd.dense_from_cupydense),
        (CuPyDia, data.Dia, cdia.cupydia_from_dia),
        (data.Dia, CuPyDia, cdia.dia_from_cupydia),
        (CuPyDia, CuPyDense, cdia.cupydia_from_cupydense),
    ]
)
data.to.register_aliases(["cupyd"], CuPyDense)
data.to.register_aliases(["cupydia"], CuPyDia)


def is_cupydense(data):
    return isinstance(data, CuPyDense)


data.create.add_creators([(is_cupydense, CuPyDense, 80)])


data.adjoint.add_specialisations([
    (CuPyDense, CuPyDense, cd.adjoint_cupydense),
    (CuPyDia, CuPyDia, cdia.adjoint_cupydia),
])
data.transpose.add_specialisations([
    (CuPyDense, CuPyDense, cd.transpose_cupydense),
    (CuPyDia, CuPyDia, cdia.transpose_cupydia),
])
data.conj.add_specialisations([
    (CuPyDense, CuPyDense, cd.conj_cupydense),
    (CuPyDia, CuPyDia, cdia.conj_cupydia),
])
data.trace.add_specialisations([
    (CuPyDense, cd.trace_cupydense),
    (CuPyDia, cdia.trace_cupydia),
])
data.mul.add_specialisations([
    (CuPyDense, CuPyDense, cd.mul_cupydense),
    (CuPyDia, CuPyDia, cdia.mul_cupydia),
])
data.imul.add_specialisations([
    (CuPyDense, CuPyDense, cd.imul_cupydense),
    (CuPyDia, CuPyDia, cdia.imul_cupydia),
])
data.neg.add_specialisations([
    (CuPyDense, CuPyDense, cd.neg_cupydense),
    (CuPyDia, CuPyDia, cdia.neg_cupydia),
])
data.matmul.add_specialisations([
    (CuPyDense, CuPyDense, CuPyDense, cd.matmul_cupydense),
    (CuPyDia, CuPyDia, CuPyDia, cdia.matmul_cupydia),
    (CuPyDia, CuPyDense, CuPyDense, linalg.matmul_cupydia_cupydense_cupydense),
])
data.add.add_specialisations([
    (CuPyDense, CuPyDense, CuPyDense, cd.add_cupydense),
    (CuPyDia, CuPyDia, CuPyDia, cdia.add_cupydia),
])
data.sub.add_specialisations([
    (CuPyDense, CuPyDense, CuPyDense, cd.sub_cupydense),
    (CuPyDia, CuPyDia, CuPyDia, cdia.sub_cupydia),
])
# constructor
data.diag.add_specialisations([
    (CuPyDense, cd.diags),
    (CuPyDia, cdia.diags),
])
data.identity.add_specialisations([
    (CuPyDense, cd.identity),
    (CuPyDia, cdia.identity),
])
data.zeros.add_specialisations([
    (CuPyDense, cd.zeros),
    (CuPyDia, cdia.zeros),
])

# dense_functions
data.tidyup.add_specialisations([(CuPyDense, cdf.tidyup_dense)])

# Why 2 trace function?
#data.trace.add_specialisations([
#    (CuPyDense, cdf.trace_cupydense)
#])

data.reshape.add_specialisations([
    (CuPyDense, CuPyDense, cdf.reshape_cupydense),
    # (CuPyDia, CuPyDia, cdiaf.reshape_cupydia),
])
data.column_stack.add_specialisations([
    (CuPyDense, CuPyDense, cdf.column_stack_cupydense),
])
data.column_unstack.add_specialisations([
    (CuPyDense, CuPyDense, cdf.column_unstack_cupydense),
])
data.split_columns.add_specialisations([
    (CuPyDense, cdf.split_columns_cupydense),
])

data.inner.add_specialisations([
    (CuPyDense, CuPyDense, cdf.inner_cupydense),
])
data.inner_op.add_specialisations([
    (CuPyDense, CuPyDense, CuPyDense, cdf.inner_op_cupydense),
    (CuPyDense, CuPyDia, CuPyDense, cdiaf.inner_op_cupydense_dia_dense),
])
data.kron.add_specialisations([
    (CuPyDense, CuPyDense, CuPyDense, cdf.kron_cupydense),
    # (CuPyDia, CuPyDia, CuPyDia, cdiaf.kron_cupydia),
])

data.norm.l2.add_specialisations([
    (CuPyDense, cdf.l2_cupydense),
    (CuPyDia, cdiaf.l2_cupydia),
])
data.norm.frobenius.add_specialisations([
    (CuPyDense, cdf.frobenius_cupydense),
    (CuPyDia, cdiaf.frobenius_cupydia),
])
data.norm.max.add_specialisations([
    (CuPyDense, cdf.max_cupydense),
    (CuPyDia, cdiaf.max_cupydia),
])
data.norm.one.add_specialisations([
    (CuPyDense, cdf.one_cupydense),
    (CuPyDia, cdiaf.one_cupydia),
])

data.inv.add_specialisations([(CuPyDense, CuPyDense, linalg.inv_cupydense)])
data.pow.add_specialisations([(CuPyDense, CuPyDense, cdf.pow_cupydense)])
data.project.add_specialisations([(CuPyDense, CuPyDense, cdf.project_cupydense)])


data.isherm.add_specialisations([
    (CuPyDense, cdf.isherm_cupydense),
    (CuPyDia, cdiaf.isherm_cupydia),
])


# We must register the functions to the data layer but do not want
# the data layer or qutip_cupy.dense to be accessible from qutip_cupy
del data
del cd
