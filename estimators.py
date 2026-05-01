from typing import Callable

import numpy as np
from scipy.signal import convolve

from spectrumlab.grids import Grid, InterpolationKind, integrate_grid
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, Number, R


def create_matched_filter(
    template: Array[R],
) -> Callable[[Array[R]], Array[R]]:
    template = template / np.linalg.norm(template)

    def transformer(intensity: Array[R]) -> Array[R]:
        return convolve(
            intensity,
            template[None, :],
            mode='same',
        )
    return transformer


def estimate_by_amplitude(
    spectrum: Spectrum,
    position: Number,
    transformer: Callable[[Array[R]], Array[R]] | None = None,
) -> Array[R]:

    if transformer:
        intensity = transformer(spectrum.intensity)
    else:
        intensity = spectrum.intensity

    score = intensity[:, position]
    return score


def estimate_by_integral(
    spectrum: Spectrum,
    position: Number,
    interval: Number,
    kind: InterpolationKind = InterpolationKind.LINEAR,
) -> Array[R]:

    weight = np.zeros(spectrum.n_numbers)
    for n in spectrum.number:
        basis = np.zeros(spectrum.n_numbers)
        basis[n] = 1.0

        weight[n] = integrate_grid(
            grid=Grid(x=spectrum.number, y=basis, units=Number),
            position=position,
            interval=interval,
            kind=kind,
        )

    score = np.dot(spectrum.intensity, weight)
    return score / interval


def estimate_by_matched_filter(
    spectrum: Spectrum,
    template: Array[R],
) -> Array[R]:

    score = np.dot(spectrum.intensity, template) / np.dot(template, template)
    return score