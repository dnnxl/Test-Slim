class AugMethod:
    NO_AUGMENTATION = 'no_augmentation'
    RAND_AUGMENT = 'rand_augmentation'

class MainMethod:
    ALONE = 'samAlone'
    FEWSHOT_1_CLASS = 'fewshot1'
    FEWSHOT_2_CLASSES = 'fewshot2'
    FEWSHOT_OOD = 'fewshotOOD'
    FEWSHOT_2_CLASSES_RELATIONAL_NETWORK = 'fewshotRelationalNetwork'
    FEWSHOT_2_CLASSES_MATCHING = 'fewshotMatching'
    FEWSHOT_2_CLASSES_BDCSPN = 'fewshotBDCSPN'
    FEWSHOT_MAHALANOBIS = 'fewshotMahalanobis'
    FEWSHOT_SUBSPACES = 'fewshotSubspaces'
    FEWSHOT_2_CLASSES_PTMAP = 'fewshotPTMap'
    SELECTIVE_SEARCH = 'ss'

class SamMethod:
    SAM = 'sam'
    MOBILE_SAM = 'mobilesam'
    FAST_SAM = 'fastsam'
    EDGE_SAM = 'edgsam'
    SAM_HQ = 'samhq'

class DimensionalityReductionMethod:
    SVD = 'svd'
    PCA = 'pca'