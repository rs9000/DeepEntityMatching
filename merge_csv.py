from ntm import NLP
import pandas as pd
import torch
import recordlinkage
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source1', type=str, default="f1_parse.csv",
                        help='Source file 1', metavar='')
    parser.add_argument('--source2', type=str, default="f2_parse.csv",
                        help='Source file 2', metavar='')
    parser.add_argument('--output_file', type=str, default="clean_table.csv",
                        help='Output file', metavar='')
    parser.add_argument('--separator', type=str, default=';',
                        help='Char separator in CSV source files', metavar='')
    parser.add_argument('--n_attrs', type=int, default=5,
                        help='Attributes in sources files', metavar='')
    parser.add_argument('--blocking_size', type=int, default=5,
                        help='Number of words in answers dictionary', metavar='')
    parser.add_argument('--blocking_attr', type=str, default='title',
                        help='Number of words in answers dictionary', metavar='')
    parser.add_argument('--word_embed', type=str, default='glove.6B.50d.txt',
                        help='Word embedding file (es. GloVe)', metavar='')
    parser.add_argument('--word_embed_size', type=int, default=50,
                        help='word embedding vector size', metavar='')
    parser.add_argument('--load_model', type=str, default='checkpoint.pt',
                        help='load model file', metavar='')
    return parser.parse_args()


def merge(model, df1, df2, candidate_links):

    print("Start Merging... (it may take a while)")
    duplicates = []

    # Find duplicates
    for i in range(len(candidate_links)):
        ix1 = candidate_links[i][0]
        ix2 = candidate_links[i][1]

        is_match = model(df1.values[ix1], df2.values[ix2])
        _, is_match = torch.max(is_match, 1)
        if int(is_match):
            duplicates.append(ix1)

    print("Found: " + str(len(duplicates)) + " duplicates")
    # Delete duplicates from table1
    df1 = df1.drop(duplicates)
    print("Merging files....")

    # Merge tables and write file
    clean_table = pd.concat([df1, df2], sort=False)
    clean_table.to_csv(args.output_file, sep=args.separator, index=False)
    print("Done!")
    print("File created: " + args.output_file)


if __name__ == "__main__":
    args = parse_arguments()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df1 = pd.read_csv(args.source1, args.separator)
    df2 = pd.read_csv(args.source2, args.separator)

    indexer = recordlinkage.Index()
    indexer.sortedneighbourhood(args.blocking_attr, window=args.blocking_size)
    candidate_links = indexer.index(df1, df2)

    model = NLP(args.word_embed, args.word_embed_size, args.n_attrs, device).to(device)
    model.load_state_dict(torch.load(args.load_model))

    merge(model, df1, df2, candidate_links)
