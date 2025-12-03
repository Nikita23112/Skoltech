from rpy2.robjects.vectors import StrVector
import pandas as pd
import numpy as np
import re
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import warnings
import traceback
from rpy2 import robjects


def clean_feature_names(names):
    """Преобразует имена столбцов в формат, безопасный для R."""
    cleaned_names = [name.replace('.', '_').replace(
        '-', '_').replace(' ', '_') for name in names]
    return cleaned_names


def r_randomforest_importance(X, y, data_path=None, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features='sqrt'):
    try:
        base = importr('base')
        utils = importr('utils')
        randomForest = importr('randomForest')

        original_features = list(X.columns)
        cleaned_features = clean_feature_names(original_features)

        df_for_r = X.copy()
        df_for_r.columns = cleaned_features
        df_for_r['target'] = y.values

        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter) as cv:
            r_df = robjects.conversion.py2rpy(df_for_r)

        formula_str = "target ~ " + " + ".join(cleaned_features)
        formula = robjects.Formula(formula_str)

        print(f"Training R Random Forest with formula: {formula_str}")

        n_samples = X.shape[0]
        n_features = len(cleaned_features)
        if max_features == 'sqrt':
            max_features = round(np.sqrt(n_features))
        else:
            max_features = round(n_features * max_features)

        if max_depth is None:
            maxnodes = robjects.NULL
        else:
            maxnodes=2**max_depth

        rf_result = randomForest.randomForest(
            formula,
            data=r_df,
            ntree=n_estimators,
            nodesize=min_samples_leaf,
            maxnodes=maxnodes,
            mtry=max_features,
            importance=True
        )

        importance_r = randomForest.importance(rf_result, type=2, scale=False)

        importance_matrix = np.array(importance_r)

        feature_names_r = list(robjects.r['rownames'](importance_r))

        print(f"Importance matrix shape: {importance_matrix.shape}")
        print(f"Feature names from R: {feature_names_r}")

        if importance_matrix.ndim == 2:
            importance_values = importance_matrix[:, 0]
        else:
            importance_values = importance_matrix

        importance_series = pd.Series(
            importance_values, index=original_features)

        print("Successfully computed variable importance")
        print(importance_series)

        return importance_series.sort_values(ascending=False)

    except Exception as e:
        warnings.warn(f"R implementation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def r_ranger_importance_air(X, y, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features='sqrt'):
    try:
        base = importr('base')
        ranger = importr('ranger')

        original_features = list(X.columns)

        cleaned_features = clean_feature_names(original_features)

        df_for_r = X.copy()
        df_for_r.columns = cleaned_features

        df_for_r['target'] = y.astype(str).values

        name_map = dict(zip(cleaned_features, original_features))

        with robjects.conversion.localconverter(robjects.default_converter + pandas2ri.converter):
            r_df = robjects.conversion.py2rpy(df_for_r)

        r_target = base.factor(r_df.rx2('target'))

        r_cleaned_features = StrVector(cleaned_features)

        r_df_features_only = r_df.rx(True, r_cleaned_features)

        r_df = base.cbind(r_df_features_only, target=r_target)

        formula_str = "target ~ " + " + ".join(cleaned_features)
        formula = robjects.Formula(formula_str)

        print(f"Training R Ranger Forest with formula: {formula_str}")

        target_type = robjects.r['class'](r_df.rx2('target'))
        target_levels = robjects.r['levels'](r_df.rx2('target'))
        print(f"Target type in R: {list(target_type)}")
        print(f"Target levels: {list(target_levels)}")

        n_features = len(cleaned_features)
        if max_features == 'sqrt':
            mtry_val = round(np.sqrt(n_features))
        elif isinstance(max_features, float):
            mtry_val = round(n_features * max_features)
        else:
            mtry_val = max_features

        if max_depth is None:
            max_depth = robjects.NULL

        rf_result = ranger.ranger(
            formula,
            data=r_df,
            num_trees=n_estimators,
            min_node_size=min_samples_leaf,
            mtry=mtry_val,
            importance="impurity_corrected",
            max_depth=max_depth,
            classification=True
        )

        task_type = rf_result.rx2('treetype')[0]
        print(f"Ranger task type: {task_type}")

        importance_r = rf_result.rx2('variable.importance')
        importance_values = np.array(importance_r)
        feature_names_r = list(robjects.r['names'](importance_r))

        importance_series = pd.Series(importance_values, index=feature_names_r)
        importance_series.index = importance_series.index.map(name_map)

        print("Successfully computed variable importance for CLASSIFICATION")

        return importance_series.sort_values(ascending=False)

    except Exception as e:
        warnings.warn(f"Ranger implementation failed: {e}")
        traceback.print_exc()
        return None
