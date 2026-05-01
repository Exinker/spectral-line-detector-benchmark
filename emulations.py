import numpy as np

from spectrumlab_emulations.apertures import Aperture, RectangularApertureShape
from spectrumlab_emulations.apparatus import Apparatus, VoigtApparatusShape
from spectrumlab_emulations.devices import Device
from spectrumlab_emulations.emulations import (
    EmittedSpectrumEmulation,
    EmittedSpectrumEmulationConfig,
    SpectrumConfig,
    fetch_emulation,
)
from spectrumlab.spectra import Spectrum
from spectrumlab.types import C, Number

from configs import (
    DETECTOR,
    N_NUMBERS,
    N_TIMES,
)


EMULATION = fetch_emulation(
    config=EmittedSpectrumEmulationConfig(
        device=Device.GRAND2_I,
        detector=DETECTOR,
        line=None,
        apparatus=Apparatus(
            detector=DETECTOR,
            shape=VoigtApparatusShape(width=25, asymmetry=0, ratio=0.1),
        ),
        aperture=Aperture(
            detector=DETECTOR,
            shape=RectangularApertureShape(),
        ),
        spectrum=SpectrumConfig(
            n_numbers=N_NUMBERS,
            n_frames=1,
        ),
        concentration_ratio=1,
        background_level=0,
    ),
)


def generate_spectrum(
    emulation: EmittedSpectrumEmulation,
    position: Number,
    concentration: C,
    is_noised: bool = True,
    is_clipped: bool = True,
    random_state: int | None = None,
) -> Spectrum:

    emulation = emulation.setup(
        position=np.full(N_TIMES, position, dtype=float),
        concentration=concentration,
    )

    spectrum = emulation.run(
        is_noised=is_noised,
        is_clipped=is_clipped,
        random_state=random_state,
    )
    return spectrum
