# sottek-hearing-model
An implementation of the psychoacoustic sound quality metrics from the Sottek Hearing Model defined in ECMA-418-2.

## How to install
The package can be installed with pip:

```bash
pip install sottek-hearing-model
```

## How to use
After installing, the package functions can be imported as follows:

```python
from sottek_hearing_model import (shm_tonality_ecma,
                                  shm_loudness_ecma,
                                  shm_roughness_ecma)
```

or (since the package currently comprises a small number of functions):

```python
from sottek_hearing_model import *
```

The functions can be used to analyse 1- or 2-channel audio signals, for example:

```python
import soundfile as sf
from sottek_hearing_model import shm_tonality_ecma

audiodata, samplerate = sf.read('your_audio_file.wav')

tonality = shm_tonality_ecma(p=audiodata, samp_rate_in=samplerate,
                             axis=0, soundfield='free_frontal',
                             out_plot=True)
```

The above code will import the audio file, analyse it using the Sottek Hearing Model tonality metric, and output the results into a dict object `tonality`, as well as produce a figure plotting the results (`out_plot=True`).

By default, a progress bar is displayed illustrating the analysis computation progression. When analysing a batch of files, you may want to suppress this using `wait_bar=False`, as well as omit output plotting (the default setting is `out_plot=False`).

For more information on input arguments, use the help docstrings.

Since the Sottek Hearing Model loudness metric uses the tonal and noise loudness components that are generated using the tonality algorithms, a convenience function, `shm_loudness_ecma_from_comp()` is provided, which reduces the loudness computation time to negligible when also calculating the tonality metric. The use of this convenience function is demonstrated in the next example below. When using this convenience function, the soundfield type cannot be specified, as this is inherited from the component loudnesses.

The corresponding reference signals for each metric, which are used for calibration and testing, can be generated and analysed as follows (which also demonstrates the `shm_loudness_ecma_from_comp()` convenience function):

```python
from sottek_hearing_model import *

(sine_1k_40dB,
 sine_1kHz_70Hz_60dB,
 sine_1kHz_4Hz_60dB) = shm_generate_ref_signals(10)

tonality = shm_tonality_ecma(p=sine_1k_40dB,
                             samp_rate_in=48e3,
                             soundfield='free_frontal',
                             out_plot=True)

loudness = shm_loudness_ecma_from_comp(tonality['spec_tonal_loudness'],
                                       tonality['spec_noise_loudness'],
                                       out_plot=True)

roughness = shm_roughness_ecma(p=sine_1kHz_70Hz_60dB,
                               samp_rate_in=48e3,
                               soundfield='free_frontal',
                               out_plot=True)
```

The third reference signal generated above (`sine_1kHz_4Hz_60dB`) corresponds with the fluctuation strength metric. This metric will be added to the package in a future release.

## How to cite
The algorithms in this package were initially translated to Python from the MATLAB codes published alongside the following paper: 

> Lotinga, M. J. B., Torjussen, M., & Felix Greco, G. (2025). Verified implementations of the Sottek psychoacoustic Hearing Model standardised sound quality metrics (ECMA-418-2 loudness, tonality and roughness). Proceedings of Forum Acusticum / Euronoise, Malaga, Spain, 23–26 June 2025. [https://www.researchgate.net/publication/392904348](https://www.researchgate.net/publication/392904348)

<!---
Bibtex:
```
@inproceedings{RN14112,
   author = {Lotinga, Michael J. B. and Torjussen, Matt and Felix Greco, G.},
   title = {Verified implementations of the Sottek psychoacoustic Hearing Model standardised sound quality metrics (ECMA-418-2 loudness, tonality and roughness)},
   booktitle = {Proceedings of Forum Acusticum 2025},
   publisher = {European Acoustics Association},
   url = {https://www.researchgate.net/publication/392904348},
   year = {2025},
   howpublished = {Forum Acusticum / Euronoise, Malaga, Spain, 23–26 June 2025}
}
```
--->

## Acknowledgements
This package was developed during research undertaken as part of the RefMap project ([https://www.refmap.eu](https://www.refmap.eu)), funded by UK Research and Innovation / EU HORIZON (grant 10061935).

These implementations first originated in a MATLAB code SottekTonality.m authored by Matt Torjussen, which implemented the ECMA-418-2:2020 tonality algorithms. The code was developed and amended by Mike Lotinga with permission, who later incorporated the loudness and roughness metrics. Gil Felix Greco contributed verification testing routines that led to further improvement of the accuracy of the metrics.

The MATLAB implementations are also available as part of SQAT (Sound Quality Analysis Toolbox): [https://github.com/ggrecow/SQAT](https://github.com/ggrecow/SQAT).

## Licensing
This work is licensed under the copyleft [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html).

## Contact
If you would like to report a bug, make suggested improvements or ask a question, please open an issue on GitHub. If you would like to contribute, you could raise a pull request. For anything else, please contact Mike Lotinga ([https://github.com/mlotinga](https://github.com/mlotinga)).
