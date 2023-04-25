# Pipeline-Processing-TCGA-Slides-for-MIL

This repo provides an exhaustive pipeline of processing TCGA Whole-Slide Images for downstream Multiple Instance Learning (MIL), basically developed from CLAM (https://github.com/mahmoodlab/CLAM). 

Feel free to post your issue in this repo if you encounter any problems with this repo.

## Introduction

What this repo could help you in analyzing Whole-Slide Images with MIL is as follows:
- S01: [Downloading slides from TCGA websites](./S01-Downloading-Slides-from-TCGA.ipynb). It shows you, step by step, how to obtain the data you desire from TCGA. 
- S02: [Reorganizing slides at patient-level](./S02-Reorganizing-Slides-at-Patient-Level.ipynb). It provides the code for organizing slides at patient-level, and show you how to get the useful label data (slide-level) that you would possibly utilize in downstream MIL tasks. 
- S03: [Segmenting and patching slides](./S03-Segmenting-and-Patching-Slides.ipynb). It shows you the complete procedures and results of tissue segmentation and patching, based on an efficient preprocessing tool CLAM. Also, the basic knowledge of WSI structure is given.
- S04: [Extracting patch features](./S04-Extracting-Patch-Features.ipynb). It shows you the complete procedures and results of patch feature extraction, also based on the CLAM. 

## Acknowledgement
We thank CLAM's team [1] for contributing such an efficient and easy-to-use repo for WSI preprocessing, and TCGA [2] for making WSI data publicly-available to facilitate cancer research.

## Reference
- [1] Lu, M. Y.; Williamson, D. F.; Chen, T. Y.; Chen, R. J.; Barbieri, M.; and Mahmood, F. 2021. *Data-efficient and weakly supervised computational pathology on whole-slide images*. **Nature biomedical engineering**, 5(6): 555–570.
- [2] Kandoth, C.; McLellan, M. D.; Vandin, F.; Ye, K.; Niu, B.; Lu, C.; Xie, M.; Zhang, Q.; McMichael, J. F.; Wycza- lkowski, M. A.; Leiserson, M. D. M.; Miller, C. A.; Welch, J. S.; Walter, M. J.; Wendl, M. C.; Ley, T. J.; Wilson, R. K.; Raphael, B. J.; and Ding, L. 2013. *Mutational landscape and significance across 12 major cancer types*. **Nature**, 502: 333–339.
