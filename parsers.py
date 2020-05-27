import argparse

def parse_args():
    """Set up the various command line parameters."""

    parser = argparse.ArgumentParser(description="Image similarity")

    parser.add_argument(
        "-i",
        "--inputdir",
        help="The directory to search for images",
        default="~/Pictures",
    )

    parser.add_argument(
        "-labelcsv",
        "--labelcsvdir",
        help="The directory to load evalueation csv",
        default="./output/label_train.csv",
    )

    parser.add_argument(
        "-extractcsv",
        "--extractcsvdir",
        help="The directory to output the features extraction csv",
        default="./output/features_extraction.csv",
    )

    parser.add_argument(
        "-o",
        "--outputdir",
        help="The directory to write output htmls files",
        default="./output",
    )

    parser.add_argument(
        "-e",
        "--extractmode",
        help="The mode to extract image features",
        choices=["features", "tsne", "umap"],
        default="tsne",
    )

    parser.add_argument(
        "-m",
        "--model",
        help="The model to use to extract features.",
        choices=["resnet50", "vgg16", "xception", "vgg19", "inceptionv3", "mobilenet"],
        default="vgg16",
    )

    parser.add_argument(
        '-f',
        '--featurecols',
        help='[tsne/umap]: Numerical data column indices to treat as features. Ex: "B,C,F", use "all" to consider all columns (excluding optional unique-col).',
        default="all",
    )

    parser.add_argument(
        '-u',
        '--uniquecol',
        help='[tsne/umap]: The column index containing unique IDs for each row (typically "ID" or "Name" column). Not required. Omitted from "all" feature-cols',
        default="A",
        )

    parser.add_argument(
        '-r',
        '--reduce',
        help='[tsne/umap]: How many dimensions to reduce features to. Default is 2.',
        default='2'
    )

    parser.add_argument(
        '-c',
        '--clustermode',
        help='The method to cluster the data',
        choices=["aroc", "kmeans"],
        default='aroc'
    )

    return parser.parse_args()

def mainParser():
    parser = parse_args()
    input_dir = parser.inputdir
    label_csv_dir = parser.labelcsvdir
    extract_csv_dir = parser.extractcsvdir
    output_dir = parser.outputdir
    extract_mode = parser.extractmode
    model_name = parser.model
    feature_cols = parser.featurecols
    unique_col = parser.uniquecol
    reduce = parser.reduce
    cluster_mode = parser.clustermode


    return input_dir,label_csv_dir,extract_csv_dir,output_dir,extract_mode,model_name,feature_cols,unique_col,reduce,cluster_mode
