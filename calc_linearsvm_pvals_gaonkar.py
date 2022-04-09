"""
Computes an SVM classification between two classes and then uses the analytical approximation of Gaonkar et al. 2015
(doi: https://dx.doi.org/10.1016/j.media.2015.06.008) to calculate a p-value for each weight.
"""

import numpy as np
from scipy import stats


def get_C(X):
    """
    Calculates the C matrix (equation 4 in the paper). Splits the compuation in subparts to make it (slightly) better
    tractable

    :param X:   data matrix (numpy.array, (m, d)) (m=num_subjects, d=num_dimensions)
    :return:    C (numpy.array, (d, m))
    """
    J = np.ones((X.shape[0], 1))
    cov = np.dot(X, X.T)
    inverted_cov = np.linalg.pinv(cov)
    XXTinv_J = np.dot(inverted_cov, J)
    minus_JT_XXTinv_J_inv = np.linalg.pinv(np.dot(np.dot(-J.T, inverted_cov), J))
    JT_XXTinv = np.dot(J.T, inverted_cov)
    return np.dot(X.T, inverted_cov + np.dot(np.dot(XXTinv_J, minus_JT_XXTinv_J_inv), JT_XXTinv))


def get_sigma_2(C, rho):
    """
    Calculates the variance (equation 5 in the paper) based on C and rho
    :param C:   matrix (numpy.array) (d, m) (d=num_dimensions, m=num_subjects)
    :param rho: proportion of the positive class in the classification (i.e. sum(y==1)/y.size)
    :return:    variance for each component (numpy.array (d, ))
    """
    return (4. * rho - 4 * rho**2) * np.sum(C**2, axis=1)


def get_sj(w):
    """
    Margin-informed statistic (equation 8 in the paper). Based on the true (unpermuted) weights w
    :param w:   weights W from the SVM (numpy.array, (d, ))
    :return:    sj (numpy.array, (d, ))
    """
    return w/np.dot(w, w)


def get_zscore(sigma_2, sj):
    """
    Transforms s_j into standard normal distribution (N(0, 1); equation 15 in the paper)
    :param sigma_2: variance vector (numpy.array, (d, ))
    :param sj:      margin-informed statistic (numpy.array, (d, ))
    :return:        standardized sj (numpy.array, (d, ))
    """
    normalization_factor = np.sum(sigma_2)
    return (normalization_factor/np.sqrt(sigma_2)) * sj


def get_pval(z_scores):
    """
    Calculates p-values for z-scored values. 1 - cdf(abs(z-score)) * 2 (*2 because of the two-tailed test).
    :param z_scores:    z-scores s_j values (numpy.array, (d, ))
    :return:            p-values corresponding to the z-scores (numpy.array, (d, ))
    """
    return (1 - stats.norm.cdf(np.abs(z_scores))) * 2


def get_rho(y):
    """
    Returns the proportion of the positive class (0, 1)
    :param y:   classification labels (positive class is assumed to be labeled as 1)
    :return:    proportion of positive class
    """
    return np.sum(y == 1, dtype=np.float)/y.size


def calculate_p_values(X, y, w):
    """
    Based on the data (X), labels (y) and the trained weights of the SVM (w) calculates analytically the p-values for
    each weight
    :param X:   data matrix (numpy.array, (m, d); (m=num_subjects, d=num_dimensions))
    :param y:   labels  (numpy.array (m, ))
    :param w:   SVM weights (numpy.array (d, ))
    :return:
    """
    rho = get_rho(y)
    C = get_C(X)
    sigma_2 = get_sigma_2(C, rho)
    sj = get_sj(w)
    sj_sign = np.sign(sj)
    sj_zscored = get_zscore(sigma_2, sj)
    p = get_pval(sj_zscored)
    # to prevent problem when taking the log10 later we will set p = 0 values to the minimal value of the system
    p[p == 0] = np.finfo(np.float64).eps
    return p, sj_sign
