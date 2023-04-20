# Explanation of PCA

## Overview
- As a part of our experimentation with SWAY2's implementation, we performed Principal Component Analysis on the dataset prior to its use in SWAY.
- To implement this we imported `decomposition` class from `Scikit-Learn` library which implements `PCA`.

## Details of Implementation
- We call the `SwayWithPCAOptimizer` present in `models/optimizers`.
- Pre processing of the data is done :
    - The missing values present in col are `substituted with the mean of the col`.
    - `One Hot Encoding` is performed for `symbols`.
- In method `_run_pca` :
    - We normalize the cols before passing it into PCA object.
    - We apply PCA where the range of pca_columns is `[3,1]`.
- After which we call the `_sway` method on data rows iteratively .
- `_sway` method calls `_half` which divides the best with help of `alphashape` to determine the boundary points. 

## Conclusion
To enhance SWAY, a method was employed to identify the principal components of the input data. This involved reducing the dataset's dimensionality by utilizing PCA, which aids in identifying the significant factors that contribute to the determination of the most relevant data columns. Once PCA was implemented, the goal was to optimize the A and B axes of the cosine similarity by selecting the farthest point on the boundary as our A. This technique aided in decreasing SWAY's computational complexity by reducing the number of rows that the Better method would typically run for in conventional SWAY, making it more computationally efficient. The disadvantage of this approach is that we are choosing the axis with the greatest range rather than the one with the highest dispersed density. While this approach leads to significant improvements, it may fail in certain situations.
