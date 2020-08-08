# Author : Thomas Buffagni
# currently maintained by: Aman Agrawal
# Latest revision : 06/24/2020

import os
import time

# Metaphor labeling functions
from Sample.modules.cluster_module import clusteringFunction_2
from Sample.modules.darkthoughts import darkthoughtsFunction_2
from Sample.modules.datastructs.metaphor_identification import MetaphorIdentification
from Sample.modules.kmeans_abs_ratings_cosine_edit_distance import identify_metaphors_abstractness_cosine_edit_dist
from Sample.modules.new_cluster_module import newClusterModule
# Data structures
from Sample.modules.sample_functions import posFunction, lemmatizingFunction
# Candidate finding functions
from Sample.modules.sample_functions import verbNounFinder, adjNounFinder, nounNounFinder
from Sample.modules.utils import parseCommandLine, getText

if __name__ == "__main__":

    start_time = time.time()

    args = parseCommandLine()

    # Initialization
    MI = MetaphorIdentification()

    # Registering the Candidate Finders and the Metaphor Labelers
    MI.addCFinder("verbNoun", verbNounFinder)
    MI.addCFinder("adjNoun", adjNounFinder)
    MI.addCFinder("nounNoun", nounNounFinder)
    MI.addMLabeler("darkthoughts", darkthoughtsFunction_2)
    MI.addMLabeler("cluster", clusteringFunction_2)
    MI.addMLabeler('newCluster', newClusterModule)
    MI.addMLabeler("kmeans", identify_metaphors_abstractness_cosine_edit_dist)

    texts, sources, targets, labels = getText(args)

    # Loading the texts in the Metaphor Identification Object
    MI.addText(texts)

    # Step 1: Annotating the text
    MI.annotateAllTexts()
    MI.allAnnotTextAddColumn("POS", posFunction)  # Set a part-of-speech to each word of the string
    MI.allAnnotTextAddColumn("lemma", lemmatizingFunction)  # Set a lemma to each word of the string
    if args.verbose:
        print(MI.getAnnotatedText())

    # Step 2: Finding candidates
    if not args.cgenerator:
        candidatesID = args.cfinder
        if MI.isCFinder(args.cfinder):
            MI.findAllCandidates(args.cfinder)  # Call the candidate finder specified in the command line by the user
            if args.verbose:
                print(MI.getCandidates(args.cfinder))
        else:
            print('The candidate finder', args.cfinder, 'is invalid')
    else:
        MI.allCandidatesFromFile(sources, targets, labels, [i for i in range(len(sources))])
        candidatesID = 'fromFile'

    # Step 3: Labeling Metaphors
    cand_type = args.cfinder  # Corresponds to the Part-of-Speech of the candidates

    for mlabeler in args.mlabelers:
        if MI.isMLabeler(mlabeler):
            MI.labelAllMetaphorsOneGroup(mlabeler, candidatesID, cand_type,
                                         verbose=args.verbose)  # Call the metaphor labeler specified in the command line by the user
            if args.verbose:
                print(MI.getAllMetaphors())
        else:
            print('The metaphor labeler', mlabeler, 'is invalid')

    if args.csv:
        MI.resultsToCSV(args.csv)

    # CK = MI.cohenKappa(args.mlabelers[0], args.mlabelers[1], candidatesID, cand_type, verbose=False,
    #           already_labeled=True) print("Cohen's Kappa:", CK)

    if args.labelled_data:
        metaphors = []
        metaphor_targets = []
        non_metaphors = []
        non_metaphor_targets = []
        for i, text in enumerate(texts):
            metaphors_in_row = []
            meta_targets_in_row = []
            non_metaphors_in_row = []
            non_meta_targets_in_row = []
            for j in range(MI.getMetaphors(args.mlabelers[0])[i].getSize()):
                candidate = MI.getMetaphors(args.mlabelers[0])[i].getMetaphor(j)
                if candidate.getPredictedLabel():
                    metaphors_in_row.append(candidate.getSource())
                    meta_targets_in_row.append(candidate.getTarget())
                else:
                    non_metaphors_in_row.append(candidate.getSource())
                    non_meta_targets_in_row.append(candidate.getTarget())

            # temp code: uncomment to print candidates in a file instead of identified metaphors
            # if cand_type == 'nounNoun':
            #     metaphors_in_row = []
            #     meta_targets_in_row = []
            #     for j in range(MI.getCandidates('nounNoun')[i].getSize()):
            #         candidate = MI.getCandidates('nounNoun')[i].getCandidate(j)
            #         metaphors_in_row.append(candidate.getSource())
            #         meta_targets_in_row.append(candidate.getTarget())
            # temp code ends

            metaphors.append(';'.join(metaphors_in_row))
            non_metaphors.append(';'.join(non_metaphors_in_row))
            metaphor_targets.append(';'.join(meta_targets_in_row))
            non_metaphor_targets.append(';'.join(non_meta_targets_in_row))

        ld_filename = args.labelled_data.split('/')[-1][:-4]
        metaphors_filename = args.mlabelers[0] + '_' + args.cfinder + '_' + ld_filename + '_metaphors.csv'
        non_metaphors_filename = args.mlabelers[0] + '_' + args.cfinder + '_' + ld_filename + '_nonmetaphors.csv'
        save_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        with open(os.path.join(save_dir, metaphors_filename), 'w') as f:
            if len(targets):
                f.write('identified_sources, identified_targets, true_sources, true_targets, text\n')
            else:
                f.write('identified_sources, identified_targets, true_sources, text\n')
            for i, metaphor in enumerate(metaphors):
                if len(targets):
                    f.write('%s,%s,%s,%s,\"%s\"\n' % (metaphor, metaphor_targets[i], sources[i], targets[i], texts[i]))
                else:
                    f.write('%s,%s,%s,\"%s\"\n' % (metaphor, metaphor_targets[i], sources[i], texts[i]))

        with open(os.path.join(save_dir, non_metaphors_filename), 'w') as f:
            if len(targets):
                f.write('identified_sources, identified_targets, true_sources, true_targets, text\n')
            else:
                f.write('identified_sources, identified_targets, true_sources, text\n')
            for i, non_metaphor in enumerate(non_metaphors):
                if len(targets):
                    f.write('%s,%s,%s,%s,\"%s\"\n' % (non_metaphor, non_metaphor_targets[i], sources[i], targets[i], texts[i]))
                else:
                    f.write('%s,%s,%s,\"%s\"\n' % (non_metaphor, non_metaphor_targets[i], sources[i], texts[i]))

    print("--- %s seconds ---" % (time.time() - start_time))
