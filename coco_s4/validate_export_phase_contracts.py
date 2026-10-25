"""CPU-only contracts for Original S4 COCO export composition; no dataset access."""
import copy
import json
from pathlib import Path

import numpy as np

from .phase_export import (compose_tfss_utility_png, validate_accepted_calibration,
                           validate_accepted_fourier, validate_phase_config)


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'coco_s4/configs/export25k.json').read_text(encoding='utf-8'))
    validate_phase_config(config)
    invalid = copy.deepcopy(config)
    invalid['tfss']['eta'] = .8
    try:
        validate_phase_config(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('TFSS eta drift was accepted')

    clean_u8 = np.full((2, 3, 3), 128, dtype=np.uint8)
    clean = clean_u8.astype(np.float32)/255.
    support = np.asarray([[1., .5, 0.], [1., .5, 0.]], dtype=np.float32)
    delta = np.zeros_like(clean)
    delta[:, 0, :] = 8/255
    delta[:, 1, :] = -4/255
    q = np.asarray([[0., 0., 1.], [.25, .5, 0.]], dtype=np.float32)
    risk = np.asarray([[1., 1., 0.], [.5, .25, 1.]], dtype=np.float32)
    candidate, projected, weight = compose_tfss_utility_png(clean, delta, q, risk, support)

    expected = 1-(1-.85)*risk*(1-q)
    expected[support == 0] = 1.
    if not np.allclose(weight, expected, rtol=0, atol=1e-7):
        raise AssertionError('Export did not use canonical W=1-.15Rnt(1-Qt)')
    difference = np.abs(projected.astype(np.int16)-clean_u8.astype(np.int16))
    if int(difference[support == 1].max()) > 8:
        raise AssertionError('Person-exclusive pixels exceeded eight byte levels')
    if int(difference[support == .5].max()) > 4:
        raise AssertionError('Person/non-target overlap pixels exceeded four byte levels')
    if np.any(difference[support == 0] != 0):
        raise AssertionError('Pixels outside Person support changed')
    if candidate.shape != clean.shape or projected.dtype != np.uint8:
        raise AssertionError('Export projection changed shape or pixel encoding')

    bad_delta = delta.copy()
    bad_delta[0, 2, 0] = 1/255
    try:
        compose_tfss_utility_png(clean, bad_delta, q, risk, support)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonzero out-of-support Fourier delta was accepted')

    calibration = dict(images=25000, person_ground_truth=55096, selected_channel_count=24,
                       selected_channels=list(range(24)),
                       non_target_objects_by_class={str(i): 1 for i in range(1, 80)},
                       original_fourier_optimizer_steps=24*17,
                       accepted_clean_train_manifest_sha256='a'*64)
    validate_accepted_calibration(calibration)
    fourier = dict(epochs_completed=24, optimizer_steps=24*17,
                   train_manifest_sha256='a'*64, selected_channels=list(range(24)),
                   calibration_receipt_sha256='b'*64, clean_train_eligible_files_unchanged=True)
    validate_accepted_fourier(fourier, calibration)
    for field, value in (('selected_channel_count', 23), ('person_ground_truth', 55095)):
        invalid_calibration = copy.deepcopy(calibration)
        invalid_calibration[field] = value
        try:
            validate_accepted_calibration(invalid_calibration)
        except ValueError:
            pass
        else:
            raise AssertionError('Incompatible calibration was accepted: '+field)
    invalid_fourier = copy.deepcopy(fourier)
    invalid_fourier['optimizer_steps'] -= 1
    try:
        validate_accepted_fourier(invalid_fourier, calibration)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid Fourier schedule was accepted')

    print(json.dumps(dict(state='passed', dataset_images_read=0, gpu_initialized=False,
                          official_metrics_disabled=True, artifact_ingest_disabled=True,
                          tfss_formula='1-.15*Rnt*(1-Qt)',
                          accepted_calibration_contracts=3, accepted_fourier_contracts=2,
                          max_person_byte_difference=int(difference[support == 1].max()),
                          max_overlap_byte_difference=int(difference[support == .5].max()),
                          outside_nonzero_pixels=int(np.count_nonzero(difference[support == 0]))),
                     sort_keys=True))


if __name__ == '__main__':
    main()
