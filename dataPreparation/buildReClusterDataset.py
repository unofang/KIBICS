import pandas as pd

def buildReClusterDataset(features_df,re_cluster_imgs):
    re_cluster_features_arr = []
    for re_cluster_img_name in re_cluster_imgs:
        this_row = features_df.loc[features_df['ID'] == re_cluster_img_name]
        re_cluster_features_arr.append(this_row)

    return pd.concat(re_cluster_features_arr)
