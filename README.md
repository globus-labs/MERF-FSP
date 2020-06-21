# MERF-FSP

### 

This repository contains code associated with image processing for the [Combustion Synthesis Research Facility](https://www.anl.gov/node/122301).

Primary novelty is in an implementation of [Difference of Gaussians](https://en.wikipedia.org/wiki/Difference_of_Gaussians) on GPU using PyTorch primitives

![image](https://user-images.githubusercontent.com/5657668/82606318-de8d1e00-9b84-11ea-9968-30d46cc53b21.png)

Final result is to produce volume distribution for [PLIF](https://en.wikipedia.org/wiki/Planar_laser-induced_fluorescence) images 

![image](https://user-images.githubusercontent.com/5657668/81742208-bb1ff000-946d-11ea-9992-12a246982a33.png)

## Directory structure

* [nn_dog](./nn_dog) contains Difference of Gaussians implemented using PyTorch primitives for efficient GPU computation.
* [original_code](./original_code) legacy code.
* [profile](./profile) scripts for profiling.
* [simulation](./simulation) code for producing plausible synthetic data. Primarily useful for debugging.
* [sk_image](./sk_image) initial experiments and implementations using scikit-image primitives.
* [vips](./vips) initial experiments and implementations using vips primitives.

## Updates
`v0.1 (2020-05-11)` - initial availability

# Acknowledgements
Kyle Chard, Ryan Chard, Ian Foster, Marcus Schwarting, Aarthi Koripelly, Joseph A. Libera, Jakob Elias, Marius Stan, Noah Paulson

## For All Information

Unless otherwise indicated, this information has been authored by an employee or employees of the UChicago Argonne, LLC., operator of the Argonne National laboratory with the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this information. The public may copy and use this information without charge, provided that this Notice and any statement of authorship are reproduced on all copies.

While every effort has been made to produce valid data, by using this data, User acknowledges that neither the Government nor UChicago Argonne LLC. makes any warranty, express or implied, of either the accuracy or completeness of this information or assumes any liability or responsibility for the use of this information. Additionally, this information is provided solely for research purposes and is not provided for purposes of offering medical advice. Accordingly, the U.S. Government and UChicago Argonne LLC. are not to be liable to any user for any loss or damage, whether in contract, tort (including negligence), breach of statutory duty, or otherwise, even if foreseeable, arising under or in connection with use of or reliance on the content displayed on this site.

For Scientific and Technical Information Only © Copyright UChicago Argonne LLC. All Rights Reserved.
