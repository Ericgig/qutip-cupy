from qutip.solver.integrator import Integrator, IntegratorVern7, IntegratorVern9
from qutip.solver.mcsolve import MCSolver
from qutip.solver.mesolve import MESolver
from qutip.solver.sesolve import SESolver
from qutip.core import data as _data

from .state import cuState
from .qobjevo import CuQobjEvo


class CuIntegratorVern7(IntegratorVern7):
    supports_blackbox: bool = False  # No feedback support

    def __init__(self, system, options):
        self.system = CuQobjEvo(system)
        self.name = f"vern7 with cuDensity"
        super().__init__(self, self.system, options)

    def set_state(self, t, state):
        if not isinstance(state0, CuState):
            state0 = _data.to(CuState, state0)
        self._ode_solver.set_initial_value(state, t)
        self._is_set = True


class CuIntegratorVern9(IntegratorVern9):
    supports_blackbox: bool = False  # No feedback support

    def __init__(self, system, options):
        self.system = CuQobjEvo(system)
        self.name = f"vern9 with cuDensity"
        super().__init__(self, self.system, options)

    def set_state(self, t, state):
        if not isinstance(state0, CuState):
            state0 = _data.to(CuState, state0)
        self._ode_solver.set_initial_value(state, t)
        self._is_set = True


MCSolver.add_integrator(CuIntegratorVern7, "CuVern7")
MESolver.add_integrator(CuIntegratorVern7, "CuVern7")
SESolver.add_integrator(CuIntegratorVern7, "CuVern7")

MCSolver.add_integrator(CuIntegratorVern9, "CuVern9")
MESolver.add_integrator(CuIntegratorVern9, "CuVern9")
SESolver.add_integrator(CuIntegratorVern7, "CuVern9")
