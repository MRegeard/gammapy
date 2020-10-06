r"""
.. _LightCurve-temporal-model:

LightCurve Temporal Model
=======================

This model parametrises a lightCurve time model.


"""


from gammapy.modeling.models import LightCurveTemplateTemporalModel
from astropy.time import Time

time_range  = [Time("59100",format="mjd"), Time("59365",format="mjd")]
path = "$GAMMAPY_DATA/tests/models/light_curve/lightcrv_PKSB1222+216.fits"
light_curve_model = LightCurveTemplateTemporalModel.read(path)
light_curve_model.plot(time_range)