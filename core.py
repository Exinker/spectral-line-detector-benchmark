from collections.abc import Mapping, Sequence
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from tqdm.notebook import tqdm

from spectrumlab.peaks.analyte_peaks.shapes import PeakShape
from spectrumlab.spectra import Spectrum
from spectrumlab.types import Array, Number, R

from configs import DETECTOR
from emulations import emulate_spectrum, setup_emulation


def calculate_auc_score(
    score_h0: Array[float],
    score_h1: Array[float],
    fpr_intercept: Sequence[float],
    show: bool = False,
) -> tuple[float, Array[float], Array[float], Array[float]]:
    fpr_intercept = np.array(fpr_intercept)

    assert len(score_h0) == len(score_h1)

    auc_score = roc_auc_score(
        y_true=np.concatenate([np.full(len(score_h0), False), np.full(len(score_h1), True)]),
        y_score=np.concatenate([score_h0, score_h1]),
    )
    fpr, tpr, threshold = roc_curve(
        y_true=np.concatenate([np.full(len(score_h0), False), np.full(len(score_h1), True)]),
        y_score=np.concatenate([score_h0, score_h1]),
    )

    tpr_intercept = np.zeros(fpr_intercept.shape)
    threshold_intercept = np.zeros(fpr_intercept.shape)
    for i, _ in enumerate(fpr_intercept):
        tpr_intercept[i] = float(np.interp(fpr_intercept[i], fpr, tpr))
        threshold_intercept[i] = float(np.interp(fpr_intercept[i], fpr, threshold))

    if show:

        plt.plot(
            score_h0,
            alpha=.75,
            label='H0',
        )
        plt.plot(
            score_h1,
            alpha=.75,
            label='H1',
        )
        noise_read = 100 * DETECTOR.config.read_noise / DETECTOR.config.capacity
        plt.axhline(
            3 * noise_read,
            linestyle='--', color='red',
        )
        for threshold in threshold_intercept:
            plt.axhline(
                threshold,
                linestyle=':', color='black',
            )

        plt.xlabel('time')
        plt.ylabel('R')
        plt.grid(linestyle=':', color='grey')
        plt.legend(loc='upper left')
        plt.show()

    if show:
        plt.plot(
            fpr,
            tpr,
        )
        notes = [
            '',
            'AUC = {:.4f}'.format(auc_score),
            '',
        ]
        for i, _ in enumerate(fpr_intercept):
            plt.axvline(
                fpr_intercept[i],
                linestyle='--', color='red',
            )
            plt.scatter(
                [fpr_intercept[i]],
                [tpr_intercept[i]],
                color='red', s=20, zorder=5,
            )
            notes.append('({:.4f}, {:.4f})'.format(fpr_intercept[i], tpr_intercept[i]))
        plt.text(
            0.70, 0.98, '\n'.join(notes),
            transform=plt.gca().transAxes,
            ha='left', va='top',
        )
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.grid(
            linestyle=':', color='grey',
        )
        plt.show()

    return auc_score, fpr_intercept, tpr_intercept, threshold_intercept


def run_experiment(
    peak_shape: PeakShape,
    position: Number,
    amplitude: Array[R],
    estimators: Mapping[str, Callable[[Spectrum], Array[R]]],
    fpr_intercept: Sequence[float],
) -> tuple[Mapping[str, Sequence[float]], Mapping[str, Sequence[float]], Mapping[str, Sequence[float]]]:
    fpr_intercept = np.array(fpr_intercept)

    emulation = setup_emulation(
        peak_shape=peak_shape,
    )

    auc_scores = {}
    fprs = {}
    tprs = {}
    for key, estimator in tqdm(estimators.items(), desc='Estimators', position=0):

        score_h0 = estimator(
            spectrum=emulate_spectrum(
                emulation=emulation,
                position=position,
                concentration=0,
            ),
        )

        auc_score = np.zeros(amplitude.shape)
        fpr = np.zeros((amplitude.size, fpr_intercept.size))
        tpr = np.zeros((amplitude.size, fpr_intercept.size))
        for i, value in enumerate(tqdm(amplitude, desc=key, position=1, leave=False)):
            score_h1 = estimator(
                spectrum=emulate_spectrum(
                    emulation=emulation,
                    position=position,
                    concentration=emulation.transform_amplitude_to_concentration(
                        position=position,
                        amplitude=value,
                    ),
                ),
            )

            auc_score[i], fpr[i], tpr[i], _ = calculate_auc_score(
                score_h0=score_h0,
                score_h1=score_h1,
                fpr_intercept=fpr_intercept,
            )

        auc_scores[key] = auc_score
        fprs[key] = fpr
        tprs[key] = tpr

    return auc_scores, fprs, tprs


def show_experiment(
    amplitude: Array[R],
    auc_score: Sequence[float],
    fpr: Sequence[float],
    tpr: Sequence[float],
    label: str,
) -> None:

    fig, (ax_left, ax_right)= plt.subplots(ncols=2, figsize=(12, 6))

    plt.sca(ax_left)
    plt.plot(
        amplitude,
        auc_score,
        label=label,
    )
    plt.axvline(
        3 * 100 * DETECTOR.config.read_noise / DETECTOR.config.capacity,
        linestyle='--', color='red',
    )
    plt.xscale('log')
    plt.xlabel('R')
    plt.ylabel('AUC')
    plt.grid(linestyle=':', color='grey')
    plt.legend(loc='upper left')

    plt.sca(ax_right)
    plt.plot(
        amplitude,
        tpr,
        label=[
            'FPR = {:.1f}%'.format(100*level)
            for level in fpr[0]
        ],
    )
    plt.axvline(
        3 * 100 * DETECTOR.config.read_noise / DETECTOR.config.capacity,
        linestyle='--', color='red',
        label=r'$3 \cdot \sigma_{rd}$',
    )
    plt.xscale('log')
    plt.xlabel('R')
    plt.ylabel('TPR')
    plt.grid(linestyle=':', color='grey')
    plt.legend(loc='upper left')
    plt.show()
