from spectrumlab.peaks.analyte_peaks.shapes import PeakShape
from spectrumlab.types import Array, Number, R


def build_template(
    peak_shape: PeakShape,
    number: Array[Number],
    position: Number,
    intensity: float = 1,
) -> Array[R]:

    return peak_shape(
        x=number,
        position=position,
        intensity=intensity,
    )
