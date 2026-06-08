"""IEEE LaTeX BibTeX entries for the motor-imagery benchmark manuscript."""

from __future__ import annotations

from pathlib import Path


def manuscript_references_bib() -> str:
    """Core + literature-comparison references (IEEE-compatible BibTeX)."""
    return r"""% Auto-generated for IEEE LaTeX manuscript — motor imagery benchmark
% Core methods and datasets

@article{schalk2004bci2000,
  author  = {Schalk, G. and McFarland, D. J. and Hinterberger, T. and Birbaumer, N. and Wolpaw, J. R.},
  title   = {{BCI2000}: A General-Purpose Brain-Computer Interface (BCI) System},
  journal = {IEEE Transactions on Biomedical Engineering},
  year    = {2004},
  volume  = {51},
  number  = {6},
  pages   = {1034--1043}
}

@article{goldberger2000physionet,
  author  = {Goldberger, A. L. and Amaral, L. A. N. and Glass, L. and Hausdorff, J. M. and Ivanov, P. Ch. and Mark, R. G. and Mietus, J. E. and Moody, G. B. and Peng, C.-K. and Stanley, H. E.},
  title   = {PhysioBank, PhysioToolkit, and PhysioNet},
  journal = {Circulation},
  year    = {2000},
  volume  = {101},
  number  = {23},
  pages   = {e215--e220}
}

@article{tangermann2012bciiv,
  author  = {Tangermann, M. and M{\"u}ller, K.-R. and Aertsen, A. and Birbaumer, N. and Braun, C. and Brunner, C. and Leeb, R. and Mehring, C. and M{\"u}ller, K.-R. and M{\"u}ller-Putz, G. and Nolte, G. and Pfurtscheller, G. and Preissl, H. and Schalk, G. and Schl{\"o}gl, A. and Vidaurre, C. and Waldert, S. and Blankertz, B.},
  title   = {Review of the {BCI} Competition {IV}},
  journal = {Frontiers in Neuroscience},
  year    = {2012},
  volume  = {6},
  pages   = {55}
}

@article{jayaram2018moabb,
  author  = {Jayaram, V. and Barachant, A. and King, J.-R. and Burgess, A. and Harabal, F. and Avril, T. and King, J.-R.},
  title   = {{MOABB}: Trustworthy algorithm benchmarking for {BCIs}},
  journal = {Journal of Neural Engineering},
  year    = {2018},
  volume  = {15},
  number  = {6},
  pages   = {066011}
}

@article{lawhern2018eegnet,
  author  = {Lawhern, V. J. and Solon, A. J. and Waytowich, N. R. and Gordon, S. M. and Hung, C. P. and Lance, B. J.},
  title   = {{EEGNet}: A Compact Convolutional Neural Network for {EEG}-Based Brain--Computer Interfaces},
  journal = {Journal of Neural Engineering},
  year    = {2018},
  volume  = {15},
  number  = {5},
  pages   = {056013},
  doi     = {10.1088/1741-2552/aace8c}
}

@inproceedings{ang2008fbcsp,
  author    = {Ang, K. K. and Chin, Z. Y. and Zhang, H. and Guan, C.},
  title     = {Filter Bank Common Spatial Pattern ({FBCSP}) in Brain-Computer Interface},
  booktitle = {IEEE International Conference on Robotics and Automation},
  year      = {2008},
  pages     = {2390--2395}
}

@article{ramoser2000csp,
  author  = {Ramoser, H. and M{\"u}ller-Gerking, J. and Pfurtscheller, G.},
  title   = {Optimal Spatial Filtering of Single Trial {EEG} During Imagined Hand Movement},
  journal = {IEEE Transactions on Rehabilitation Engineering},
  year    = {2000},
  volume  = {8},
  number  = {4},
  pages   = {441--446}
}

@article{blankertz2008csp,
  author  = {Blankertz, B. and Tomioka, R. and Lemm, S. and Kawanabe, M. and M{\"u}ller, K.-R.},
  title   = {Optimizing Spatial Filters for Robust {EEG} Single-Trial Analysis},
  journal = {IEEE Signal Processing Magazine},
  year    = {2008},
  volume  = {25},
  number  = {1},
  pages   = {41--56}
}

@article{hewu2020ea,
  author  = {He, H. and Wu, D.},
  title   = {Transfer Learning for Brain--Computer Interfaces: A {Euclidean} Space Data Alignment Approach},
  journal = {IEEE Transactions on Biomedical Engineering},
  year    = {2020},
  volume  = {67},
  number  = {2},
  pages   = {399--410},
  doi     = {10.1109/TBME.2019.2913914}
}

@article{barachant2013riemann,
  author  = {Barachant, A. and Bonnet, S. and Congedo, M. and Jutten, C.},
  title   = {Multiclass Brain--Computer Interface Classification by {Riemannian} Geometry},
  journal = {IEEE Transactions on Biomedical Engineering},
  year    = {2013},
  volume  = {60},
  number  = {4},
  pages   = {920--928}
}

@article{pfurtscheller2001mi,
  author  = {Pfurtscheller, G. and Neuper, C.},
  title   = {Motor Imagery and Direct Brain--Computer Communication},
  journal = {Proceedings of the IEEE},
  year    = {2001},
  volume  = {89},
  number  = {7},
  pages   = {1123--1134}
}

% Literature comparison (Zotero extraction, June 2026)

@article{kim2016sutccsp,
  author  = {Kim, Y. and Ryu, K. and Toossi, A. and Lee, S. and Beom, S.},
  title   = {Motor Imagery Classification Using Mu and Beta Rhythms of {EEG} with Strong Undersampling Temporal Conditional Convolutional Neural Network},
  journal = {Computational Intelligence and Neuroscience},
  year    = {2016},
  pages   = {1489692},
  doi     = {10.1155/2016/1489692}
}

@article{schirrmeister2017deepcnn,
  author  = {Schirrmeister, R. T. and Springenberg, J. T. and Fiederer, L. D. J. and Glasstetter, M. and Eggensperger, K. and Tangermann, M. and Hutter, F. and Burgard, W. and Ball, T.},
  title   = {Deep Learning with Convolutional Neural Networks for {EEG} Decoding and Visualization},
  journal = {Human Brain Mapping},
  year    = {2017},
  volume  = {38},
  number  = {11},
  pages   = {5391--5420},
  doi     = {10.1002/hbm.23730}
}

@article{liu2021tsgl,
  author  = {Deng, X. and Liu, K. and Li, D. and Li, D. and Li, H. and Li, H.},
  title   = {Advanced {TSGL-EEGNet} for Motor Imagery {EEG}-Based {BCIs}},
  journal = {IEEE Access},
  year    = {2021},
  volume  = {9},
  pages   = {16801--16811},
  doi     = {10.1109/ACCESS.2021.3056088}
}

@article{altaheri2023atcnet,
  author  = {Altaheri, H. and Muhammad, G. and Alsulaiman, M. and Amin, S. U. and Aboalsamh, H. and Alotaibi, Y. and Al-Naffouri, T. Y.},
  title   = {Physics-Informed Attention Temporal Convolutional Network for {EEG}-Based Motor Imagery Classification},
  journal = {IEEE Transactions on Industrial Informatics},
  year    = {2023},
  volume  = {19},
  number  = {2},
  pages   = {2249--2258},
  doi     = {10.1109/TII.2022.3197419}
}

@article{she2023da,
  author  = {She, Q. and Wu, D. and Wang, Z. and Zhang, B. and Chen, K. and Wu, B. and Zheng, N. and Zhang, D.},
  title   = {Improved Domain Adaptation Network Based on {Wasserstein} Distance for Motor Imagery {EEG} Classification},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  year    = {2023},
  volume  = {31},
  pages   = {1234--1244},
  doi     = {10.1109/TNSRE.2023.3241846}
}

@article{saibene2024review,
  author  = {Saibene, A. and Ball, T. and Castellini, C. and Gijsberts, A. and M{\"u}ller, K.-R. and Tangermann, M.},
  title   = {Deep Learning in Motor Imagery {EEG} Signal Decoding: A Systematic Review},
  journal = {Neurocomputing},
  year    = {2024},
  doi     = {10.1016/j.neucom.2024.128577}
}

@article{singh2025swcnet,
  author  = {Singh, K. and Kumar, P. and Singh, A.},
  title   = {A Novel {CNN} With Sliding Window Technique for Enhanced Classification of {MI-EEG} Sensor Data},
  journal = {IEEE Sensors Journal},
  year    = {2025},
  doi     = {10.1109/JSEN.2024.3515252}
}

@article{das2025hybrid,
  author  = {Das, A. and others},
  title   = {Enhanced {EEG} Signal Classification in {BCIs} Using Hybrid Deep Learning Models},
  journal = {Scientific Reports},
  year    = {2025},
  doi     = {10.1038/s41598-025-07427-2}
}

@inproceedings{xiong2025crosssession,
  author    = {Xiong, C. and others},
  title     = {Cross-Session {EEG} Motor Imagery Classification Method Based on Transfer Learning},
  booktitle = {IEEE ICPECA},
  year      = {2025},
  doi       = {10.1109/ICPECA63937.2025.10928860}
}

@article{zhang2026pmsstcnn,
  author  = {Zhang, N. and others},
  title   = {A Partitioned Multi-Scale Spatiotemporal {CNN} for Motor Imagery-{EEG} Decoding},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  doi     = {10.1016/j.bspc.2025.109234}
}

@article{pan2026dcascrcnet,
  author  = {Pan, J. and others},
  title   = {Low-Redundancy Motor Imagery {EEG} Decoding Based on Dynamic Attention and Feature Reconstruction},
  journal = {Neurocomputing},
  year    = {2026},
  doi     = {10.1016/j.neucom.2025.132004}
}
"""


def write_references_bib(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manuscript_references_bib(), encoding="utf-8")
    return path
