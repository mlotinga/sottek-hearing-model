# sottek-hearing-model
This package is an implementation of the psychoacoustic sound quality metrics from the Sottek Hearing Model defined in ECMA-418-2.

## How to install
The package can be installed with pip:

```bash
pip install sottek-hearing-model
```

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
This package was developed during research undertaken as part of the RefMap project ([https://www.refmap.eu](https://www.refmap.eu)), funded by UK Research and Innovation / EU HORIZON.

These implementations first originated in a MATLAB code SottekTonality.m authored by Matt Torjussen, which implemented the ECMA-418-2:2020 tonality algorithms. The code was developed and amended by Mike Lotinga with permission, who later incorporated the loudness and roughness metrics.

The MATLAB implementations are also available as part of SQAT (Sound Quality Analysis Toolbox): [https://github.com/ggrecow/SQAT](https://github.com/ggrecow/SQAT)

## Licensing
This work is licensed under the copyleft GNU General Public License v3.

## Contact
If you would like to report a bug, make suggested improvements or ask a question, please open an issue on GitHub. If you would like to contribute, you could raise a pull request. For anything else, please contact Mike Lotinga ([https://github.com/mlotinga](https://github.com/mlotinga)).




