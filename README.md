# Pipeline-Processing-TCGA-Slides-for-MIL

[[Overview]](https://github.com/liupei101/Pipeline-Processing-TCGA-Slides-for-MIL?tab=readme-ov-file#overview) | [[Walkthrough]](https://github.com/liupei101/Pipeline-Processing-TCGA-Slides-for-MIL?tab=readme-ov-file#walkthrough) | [[Acknowledgement]](https://github.com/liupei101/Pipeline-Processing-TCGA-Slides-for-MIL?tab=readme-ov-file#acknowledgement) | [[Citation]](https://github.com/liupei101/Pipeline-Processing-TCGA-Slides-for-MIL?tab=readme-ov-file#citation)

📚 Recent updates:
- 24/12/18: add a new feature extractor `CONCH_v1.5` from [`TITAN`](https://github.com/mahmoodlab/TITAN)
- 24/12/06: add a new feature extractor `UNI`
- 24/06/01: add new features: allowing users to specify **a unified magnification** for patching and feature extraction
- 24/05/08: update codes & readme

## Overview

This repo provides **a complete and detailed tutorial** for users who intend to **process TCGA Whole-Slide Images (WSIs) for downstream computational tasks**, such as WSI classification and survival analysis (basically with multiple-instance learning (MIL) as the learning paradigm). 

🔥 Moreover, this repo also provides [an improved version](./tools/CLAM) of original [CLAM](https://github.com/mahmoodlab/CLAM). This improvoed version has more practical features: 
- allowing users to **specify a unified magnification** for patching and feature extractiong,
- **patch-level image transformations** for model rubostness test, e.g. Gaussian bluring and stain variation.
- **diverse architectures / networks** for patch feature extracting

💡 A quick overview of what convenience this repo could provide for you:
- **Step-by-step instructions to download complete TCGA data**, including WSIs and corresponding label of interest. This is not presented in CLAM.
- **Detailed steps to process WSIs**, including WSI segmentation & patching and patch feature extraction. Two notebooks are given to show the details that need to be noticed when using CLAM for WSI preprocessing.
- **More alternative pretrained models for patch feature extraction**, currently with support of [truncated ResNet50 (CLAM)](https://github.com/mahmoodlab/CLAM), [ResNet18 w/ SimCL (DSMIL)](https://github.com/binli123/dsmil-wsi), [CTransPath](https://github.com/Xiyue-Wang/TransPath), [CLIP](https://github.com/openai/CLIP), [PLIP](https://github.com/PathologyFoundation/plip), [UNI](https://github.com/mahmoodlab/UNI), [CONCH](https://github.com/mahmoodlab/CONCH), and [CONCH_v1.5](https://github.com/mahmoodlab/TITAN).

📝 This repo is developed from [***PseMix***](https://github.com/liupei101/PseMix), previously. Now it has been moved out as an individual project for maintaining convenience. 

*On updating*. Stay tuned!

Feel free to post your issue in this repo if you encounter any problems.

## Walkthrough

👩‍💻 Here show you how to use this repo:
- S01: [Downloading slides from TCGA websites](./S01-Downloading-Slides-from-TCGA.ipynb). It shows you, step by step, how to obtain the data you want from TCGA. 
- S02: [Reorganizing slides at patient-level](./S02-Reorganizing-Slides-at-Patient-Level.ipynb). It provides the code for organizing slides at patient-level and shows you how to get the useful label data (slide-level) that you would possibly utilize in downstream computational tasks. 
- S03: [Segmenting and patching slides](./S03-Segmenting-and-Patching-Slides.ipynb). It shows you the complete procedures and results of tissue segmentation and patching. Also, the basic knowledge of WSI structure is given.
- S04: [Extracting patch features](./S04-Extracting-Patch-Features.ipynb). It shows you the complete procedures and results of patch feature extraction. 6+ pertained models are supported and can be easily used with this step. 

📝 Some notes:
- It is strongly recommended to use `CONCH`/`UNI`/`CONCH_v1.5` in patch feature extraction (S04)
- If you cannot use `CONCH`/`UNI`/`CONCH_v1.5` due to limited access rights, `CTransPath` and `PLIP` are good alternatives (both are free for use). Their usage is provided in [tools/scripts](https://github.com/liupei101/Pipeline-Processing-TCGA-Slides-for-MIL/tree/main/tools/scripts).
- The `truncated ResNet50` and `ResNet18 w/ SimCL` are **NOT** recommended in patch feature extraction. 

## Acknowledgement
We thank [CLAM team](https://github.com/mahmoodlab/CLAM) for contributing such an efficient and easy-to-use tool for WSI preprocessing, and [TCGA team](https://www.cancer.gov/ccg/research/genome-sequencing/tcga) for making WSI data publicly-available to facilitate research.

## Citation

If this project helps you more or less, please cite it via 
```txt
@article{liu10385148,
  author={Liu, Pei and Ji, Luping and Zhang, Xinyu and Ye, Feng},
  journal={IEEE Transactions on Medical Imaging}, 
  title={Pseudo-Bag Mixup Augmentation for Multiple Instance Learning-Based Whole Slide Image Classification}, 
  year={2024},
  volume={43},
  number={5},
  pages={1841-1852},
  doi={10.1109/TMI.2024.3351213}
}
```

In addition, if you paper use TCGA data and CLAM tools, please also cite them via
```txt
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y and Williamson, Drew FK and Chen, Tiffany Y and Chen, Richard J and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature biomedical engineering},
  volume={5},
  number={6},
  pages={555--570},
  year={2021},
  publisher={Nature Publishing Group UK London}
}

@article{kandoth2013mutational,
  title={Mutational landscape and significance across 12 major cancer types},
  author={Kandoth, Cyriac and McLellan, Michael D and Vandin, Fabio and Ye, Kai and Niu, Beifang and Lu, Charles and Xie, Mingchao and Zhang, Qunyuan and McMichael, Joshua F and Wyczalkowski, Matthew A and others},
  journal={Nature},
  volume={502},
  number={7471},
  pages={333--339},
  year={2013},
  publisher={Nature Publishing Group UK London}
}
```
