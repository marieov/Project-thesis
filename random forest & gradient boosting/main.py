import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import config
from importances import find_importance
from accuracy import calculate_accuracy


# (1) Feature importance
find_importance(config.FEAT_FILE_FEATURES, config.IMPORTANCE_FILE_FEATURES, config.FEATURE_NAMES, find_important_channels=False, find_important_features=True, mdi=False, mda=True)

# (2) Channel importance
find_importance(config.FEAT_FILE_CHANNELS, config.IMPORTANCE_FILE_CHANNELS, config.CHANNEL_NAMES, find_important_channels=True, find_important_features=False, mdi=True, mda=False)

# (3) Finding accuracy of models using most important features and increasing number channels (ordered after importance)
calculate_accuracy(find_important_channels=False, find_important_features=True, mdi=False, mda=True)

        
