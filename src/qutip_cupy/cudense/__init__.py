



from qutip.core.options import QutipOptions


class cuDensityOption(QutipOptions):
    _options = {
        "ctx": WorkStream()
    }
    _settings_name = "cuDensity"
    _properties = {}


cuDensityOption._set_as_global_default()
