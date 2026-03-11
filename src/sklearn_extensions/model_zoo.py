"""
Taken from the hunded-hammers package's source code at https://github.com/vgarciasc/hundred-hammers
"""

from sklearn.dummy import DummyClassifier  # pylint: disable=W0611
from sklearn.dummy import DummyRegressor  # pylint: disable=W0611

from sklearn.isotonic import IsotonicRegression  # pylint: disable=W0611

from sklearn.neighbors import KNeighborsClassifier  # pylint: disable=W0611
from sklearn.neighbors import KNeighborsRegressor  # pylint: disable=W0611

from sklearn.linear_model import LinearRegression  # pylint: disable=W0611
from sklearn.linear_model import Ridge  # pylint: disable=W0611
from sklearn.linear_model import RidgeCV  # pylint: disable=W0611
from sklearn.linear_model import Lasso  # pylint: disable=W0611
from sklearn.linear_model import LassoCV  # pylint: disable=W0611
from sklearn.linear_model import ElasticNet  # pylint: disable=W0611
from sklearn.linear_model import ElasticNetCV  # pylint: disable=W0611
from sklearn.linear_model import BayesianRidge  # pylint: disable=W0611
from sklearn.linear_model import QuantileRegressor  # pylint: disable=W0611
from sklearn.linear_model import BayesianRidge  # pylint: disable=W0611
from sklearn.linear_model import BayesianRidge  # pylint: disable=W0611
from sklearn.linear_model import LogisticRegression  # pylint: disable=W0611
from sklearn.linear_model import RidgeClassifier  # pylint: disable=W0611
from sklearn.linear_model import RidgeClassifierCV  # pylint: disable=W0611
from sklearn.linear_model import SGDClassifier  # pylint: disable=W0611
from sklearn.linear_model import SGDRegressor  # pylint: disable=W0611
from sklearn.linear_model import Perceptron  # pylint: disable=W0611
from sklearn.linear_model import PassiveAggressiveClassifier  # pylint: disable=W0611
from sklearn.linear_model import PassiveAggressiveRegressor  # pylint: disable=W0611

from sklearn.kernel_ridge import KernelRidge  # pylint: disable=W0611

from sklearn.neural_network import MLPClassifier  # pylint: disable=W0611
from sklearn.neural_network import MLPRegressor  # pylint: disable=W0611
from sklearn.neural_network import BernoulliRBM  # pylint: disable=W0611

from sklearn.svm import LinearSVC  # pylint: disable=W0611
from sklearn.svm import LinearSVR  # pylint: disable=W0611
from sklearn.svm import SVC  # pylint: disable=W0611
from sklearn.svm import SVR  # pylint: disable=W0611

from sklearn.tree import DecisionTreeClassifier  # pylint: disable=W0611
from sklearn.tree import DecisionTreeRegressor  # pylint: disable=W0611
from sklearn.tree import ExtraTreeClassifier  # pylint: disable=W0611
from sklearn.tree import ExtraTreeRegressor  # pylint: disable=W0611

from sklearn.ensemble import AdaBoostClassifier  # pylint: disable=W0611
from sklearn.ensemble import AdaBoostRegressor  # pylint: disable=W0611
from sklearn.ensemble import GradientBoostingClassifier  # pylint: disable=W0611
from sklearn.ensemble import GradientBoostingRegressor  # pylint: disable=W0611
from sklearn.ensemble import RandomForestClassifier  # pylint: disable=W0611
from sklearn.ensemble import RandomForestRegressor  # pylint: disable=W0611
from sklearn.ensemble import ExtraTreesClassifier  # pylint: disable=W0611
from sklearn.ensemble import BaggingClassifier  # pylint: disable=W0611

from sklearn.naive_bayes import BernoulliNB  # pylint: disable=W0611
from sklearn.naive_bayes import CategoricalNB  # pylint: disable=W0611
from sklearn.naive_bayes import ComplementNB  # pylint: disable=W0611
from sklearn.naive_bayes import GaussianNB  # pylint: disable=W0611
from sklearn.naive_bayes import MultinomialNB  # pylint: disable=W0611

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # pylint: disable=W0611
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # pylint: disable=W0611

from sklearn.gaussian_process import GaussianProcessClassifier  # pylint: disable=W0611
from sklearn.gaussian_process import GaussianProcessRegressor  # pylint: disable=W0611

from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MeanShift
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import KernelPCA

from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding
