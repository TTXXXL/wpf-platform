# Third-party notices

Some model implementations follow the structure of the projects below. The code in this
repository was rewritten from the papers and these references; no source files were copied.
Each model file names its reference in the module docstring.

| Model | Paper | Reference code | License |
|---|---|---|---|
| AGCRN (`models/neural/agcrn.py`) | Bai et al., "Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting", NeurIPS 2020 | https://github.com/LeiBAI/AGCRN | MIT |
| TCN (`models/neural/tcn.py`) | Bai, Kolter, Koltun, "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling", 2018 | https://github.com/locuslab/TCN | MIT |
| DLinear (`models/neural/dlinear.py`) | Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023 | https://github.com/thuml/Time-Series-Library | MIT |
| PatchTST (`models/neural/patchtst.py`) | Nie et al., "A Time Series is Worth 64 Words", ICLR 2023 | https://github.com/thuml/Time-Series-Library | MIT |
| iTransformer (`models/neural/itransformer.py`) | Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024 | https://github.com/thuml/Time-Series-Library | MIT |

The MIT license of these projects requires that their copyright and permission notice be
included with copies or substantial portions of their software. Because this repository
contains reimplementations rather than copies, the notice is reproduced here for clarity:

> Permission is hereby granted, free of charge, to any person obtaining a copy of this software
> and associated documentation files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy, modify, merge, publish,
> distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
> Software is furnished to do so, subject to the following conditions: The above copyright
> notice and this permission notice shall be included in all copies or substantial portions of
> the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

Datasets are not redistributed. Kelmarsh and Penmanshiel SCADA data (Zenodo records 16807551
and 16807304) are published by Cubico Sustainable Investments under CC BY 4.0; SDWPF is
published by Baidu for the KDD Cup 2022. Cite the original sources when using them.
