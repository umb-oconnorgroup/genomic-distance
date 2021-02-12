from argparse import ArgumentParser
import sys

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean


KGP_SAMPLES_PATH = 'kgp_samples.csv'
POPULATION_DISTANCES_PATH = 'population_distances.csv'


parser = ArgumentParser(description='Calculate Genome Ranking Score')
parser.add_argument('-e', '--embeddings', type=str, required=True,
                    help='path to npy file containing sample embeddings')
parser.add_argument('-s', '--samples', type=str, required=True,
                    help='path to npy file containing sample names in same order as embedding file')
parser.add_argument('-m', '--metric', type=str, default='euclidean',
                    help='name of distance metric to be used when computing distance (supports scipy cdist metrics and hellinger)')


def main():
    args = parser.parse_args()

    kgp_sample_df = pd.read_csv(KGP_SAMPLES_PATH)
    population_distance_df = pd.read_csv(POPULATION_DISTANCES_PATH, index_col=1)

    samples = np.load(args.samples, allow_pickle=True)
    kgp_sample_df = kgp_sample_df[kgp_sample_df.apply(lambda x: x['Sample'] in samples, axis=1)]
    embeddings = np.load(args.embeddings)

    try:
        metric = getattr(sys.modules[__name__], args.metric)
    except AttributeError:
        metric = args.metric

    distance_matrix = cdist(embeddings, embeddings, metric=metric)
    ranking_matrix = distance_matrix.argsort().argsort()
    print(ranking_score(ranking_matrix, kgp_sample_df, population_distance_df, samples))

# define the hellinger distance between two probability distributions
def hellinger(a: np.ndarray, b: np.ndarray) -> float:
    return (1 / np.sqrt(2)) * euclidean(np.sqrt(a), np.sqrt(b))

def misranked_position(ranking, floor, ceiling):
    if ranking < floor:
        return floor - ranking
    elif ranking > ceiling:
        return ranking - ceiling
    else:
        return 0

def random_misranked_position(n, floor, ceiling):
    return 1 / (2 * n) * (floor ** 2 - floor + ceiling ** 2 - (2 * n + 1) * ceiling) + (n + 1) / 2

def individual_ranking_score(rankings, population_group_assignments, population_groups, average_random_misranked_positions):
    average_misranked_positions = np.mean([misranked_position(rankings[i], population_groups[population_group_assignments[i]]['floor'], population_groups[population_group_assignments[i]]['ceiling']) for i in range(len(rankings))])
    return average_random_misranked_positions / average_misranked_positions

def ranking_score_by_population(ranking_matrix, kgp_sample_df, population_distance_df, samples, population):
    population_distances = population_distance_df.loc[population, population_distance_df.columns != 'Super Population']
    population_counts = kgp_sample_df['Population'].value_counts()
    sample_to_population_group = pd.Series(kgp_sample_df['Population'].apply(lambda x: population_distances.loc[x]).values, index=kgp_sample_df['Sample'])
    population_group_assignments = sample_to_population_group.loc[samples]
    sample_to_index_map = dict([(samples[i], i) for i in range(len(samples))])
    population_groups = {}
    for population_group in sorted(population_distances.unique()):
        populations = population_distances.loc[population_distances == population_group].index
        count = population_counts.loc[populations].sum()
        floor = int(np.sum([population_groups[population_group]['count'] for population_group in population_groups]))
        population_groups[population_group] = {
            'count': count,
            'floor': floor,
            'ceiling': floor + count
            }
    average_random_misranked_positions = np.mean([random_misranked_position(len(population_group_assignments), population_groups[assignment]['floor'], population_groups[assignment]['ceiling']) for assignment in population_group_assignments])
    return kgp_sample_df[kgp_sample_df['Population'] == population].apply(lambda x: individual_ranking_score(ranking_matrix[sample_to_index_map[x['Sample']]], population_group_assignments, population_groups, average_random_misranked_positions), axis=1)

def ranking_score(ranking_matrix, kgp_sample_df, population_distance_df, samples):
    individual_ranking_scores = []
    for population in kgp_sample_df['Population'].unique():
        individual_ranking_scores.extend(ranking_score_by_population(ranking_matrix, kgp_sample_df, population_distance_df, samples, population))
    return np.mean(individual_ranking_scores)


if __name__ == '__main__':
    main()
