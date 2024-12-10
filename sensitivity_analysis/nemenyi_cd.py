import numpy as np
import pandas as pd
from scipy.special import binom
from scipy.stats import friedmanchisquare, chi2
def ranks(data: np.array, descending=True):
    """ Computes the rank of the elements in data.
    :param data: 2-D matrix
    :param descending: boolean (default False). If true, rank is sorted in descending order.
    :return: ranks, where ranks[i][j] == rank of the i-th row w.r.t the j-th column.
    """
    s = 0 if (descending is False) else 1

    # Compute ranks. (ranks[i][j] == rank of the i-th treatment on the j-th sample.)
    if data.ndim == 2:
        ranks = np.ones(data.shape)
        for i in range(data.shape[0]):
            values, indices, rep = np.unique(
                (-1) ** s * np.sort((-1) ** s * data[i, :]), return_index=True, return_counts=True, )
            for j in range(data.shape[1]):
                ranks[i, j] += indices[values == data[i, j]] + \
                               0.5 * (rep[values == data[i, j]] - 1)
        print(np.mean(ranks,axis=0))
        return ranks
    elif data.ndim == 1:
        ranks = np.ones((data.size,))
        values, indices, rep = np.unique(
            (-1) ** s * np.sort((-1) ** s * data), return_index=True, return_counts=True, )
        for i in range(data.size):
            ranks[i] += indices[values == data[i]] + \
                        0.5 * (rep[values == data[i]] - 1)
        return ranks


def sign_test(data):
    """ Given the results drawn from two algorithms/methods X and Y, the sign test analyses if
    there is a difference between X and Y.
    .. note:: Null Hypothesis: Pr(X<Y)= 0.5
    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :return p_value: The associated p-value from the binomial distribution.
    :return bstat: Number of successes.
    """

    if type(data) == pd.DataFrame:
        data = data.values

    if data.shape[1] == 2:
        X, Y = data[:, 0], data[:, 1]
        n_perf = data.shape[0]
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of dimensions for axis 1')

    # Compute the differences
    Z = X - Y
    # Compute the number of pairs Z<0
    Wminus = sum(Z < 0)
    # If H_0 is true ---> W follows Binomial(n,0.5)
    p_value_minus = 1 - binom.cdf(k=Wminus, p=0.5, n=n_perf)

    # Compute the number of pairs Z>0
    Wplus = sum(Z > 0)
    # If H_0 is true ---> W follows Binomial(n,0.5)
    p_value_plus = 1 - binom.cdf(k=Wplus, p=0.5, n=n_perf)

    p_value = 2 * min([p_value_minus, p_value_plus])

    return pd.DataFrame(data=np.array([Wminus, Wplus, p_value]), index=['Num X<Y', 'Num X>Y', 'p-value'],
                        columns=['Results'])

def friedman_test(data):
    """ Friedman ranking test.
    ..note:: Null Hypothesis: In a set of k (>=2) treaments (or tested algorithms), all the treatments are equivalent, so their average ranks should be equal.
    :param data: An (n x 2) array or DataFrame contaning the results. In data, each column represents an algorithm and, and each row a problem.
    :return p_value: The associated p-value.
    :return friedman_stat: Friedman's chi-square.
    """

    # Initial Checking
    if type(data) == pd.DataFrame:
        data = data.values

    if data.ndim == 2:
        n_samples, k = data.shape
    else:
        raise ValueError(
            'Initialization ERROR. Incorrect number of array dimensions')
    if k < 2:
        raise ValueError(
            'Initialization Error. Incorrect number of dimensions for axis 1.')

    # Compute ranks.
    datarank = ranks(data)
    print(datarank)

    # Compute for each algorithm the ranking average.
    avranks = np.mean(datarank, axis=0)
    print(avranks)

    # Get Friedman statistics
    friedman_stat = (12.0 * n_samples) / (k * (k + 1.0)) * \
                    (np.sum(avranks ** 2) - (k * (k + 1) ** 2) / 4.0)

    # Compute p-value
    p_value = (1.0 - chi2.cdf(friedman_stat, df=(k - 1)))

    return pd.DataFrame(data=np.array([friedman_stat, p_value]), index=['Friedman-statistic', 'p-value'],
                        columns=['Results'])
data = pd.read_excel('re.xlsx').set_index('Unnamed: 0')

f2 = np.row_stack((data.iloc[:,0],data.iloc[:,1],data.iloc[:,2],
                   data.iloc[:,3],data.iloc[:,4], data.iloc[:,5], data.iloc[:,6],
                   data.iloc[:,7],data.iloc[:,8],data.iloc[:,9])).T

print(friedman_test(f2))
