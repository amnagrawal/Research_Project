# Author : Thomas Buffagni
# Latest revision : 03/26/2019

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
from Sample.modules.sample_functions import verbNounFinder, adjNounFinder
from Sample.modules.utils import parseCommandLine, getText

if __name__ == "__main__":

    start_time = time.time()

    args = parseCommandLine()

    # Initialization
    MI = MetaphorIdentification()

    # Registering the Candidate Finders and the Metaphor Labelers
    MI.addCFinder("verbNoun", verbNounFinder)
    MI.addCFinder("adjNoun", adjNounFinder)
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
        for i, text in enumerate(texts):
            metaphors_identified = []
            for j in range(MI.getMetaphors(args.mlabelers[0])[i].getSize()):
                temp = MI.getMetaphors(args.mlabelers[0])[i].getMetaphor(j).getPredictedLabel()
                if temp:
                    metaphors_identified.append(MI.getMetaphors(args.mlabelers[0])[i].getMetaphor(j).getSource())

            metaphors.append(';'.join(metaphors_identified))

        filename = args.mlabelers[0] + '_' + args.cfinder + '.txt'
        save_dir = os.path.join(os.getcwd(), 'temp')
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        filename = os.path.join(save_dir, filename)
        print(f"saving file at {filename}")
        with open(filename, 'w') as f:
            for metaphor in metaphors:
                f.write('%s\n' % metaphor)

    print("--- %s seconds ---" % (time.time() - start_time))
