#python main.py -competition -cf verbNoun -csv answers1.txt -ml "newCluster"
#python main.py -competition -cf adjNoun -csv answers2.txt -ml "newCluster"
#python /home/aman/IITC/Research_Project/Sample/modules/competition/competition_output.py

python Sample/main.py -ld data/new_metaphor_corpus.csv -cf adjNoun -ml darkthoughts
python Sample/main.py -ld data/new_metaphor_corpus.csv -cf verbNoun -ml darkthoughts
python Sample/analysis.py -ld data/new_metaphor_corpus.csv -ml darkthoughts