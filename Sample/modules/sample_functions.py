from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from .datastructs.annotated_text import AnnotatedText
from .datastructs.candidate_group import CandidateGroup
from .datastructs.candidate import Candidate
from .datastructs.metaphor_group import MetaphorGroup
from .datastructs.metaphor import Metaphor


def getWordnetPos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# Use NLTK pos_tag function
def posFunction(annotatedText):
    finalPos = []
    if annotatedText.isColumnPresent("word"):
        sentence = annotatedText.getColumn("word")
    else:
        return finalPos
    pos = pos_tag(sentence)
    for i in range(len(pos)):
        finalPos.append(pos[i][1])
    return finalPos


# Use NLTK WordNetLemmatizer function
def lemmatizingFunction(annotatedText):
    lemm = WordNetLemmatizer()
    posTags = []
    finalLem = []
    sentence = []
    if annotatedText.isColumnPresent("word"):
        sentence = annotatedText.getColumn("word")
    else:
        return finalLem
    if annotatedText.isColumnPresent("POS"):
        posTags = annotatedText.getColumn("POS")
    else:
        return finalLem

    for i in range(len(sentence)):
        curentTag = getWordnetPos(posTags[i])
        if curentTag:
            finalLem.append(lemm.lemmatize(sentence[i], curentTag))
        else:
            finalLem.append(lemm.lemmatize(sentence[i]))
    return finalLem


def testIDFunction(annotatedText):
    candidates = CandidateGroup()
    testCandidate = Candidate(annotatedText, 2, (0, 2), 6, (6, 6))
    candidates.addCandidate(testCandidate)
    return candidates


def adjNounFinder(annotatedText):
    candidates = CandidateGroup()
    POScolumn = annotatedText.getColumn("POS")
    candidate = []
    currentAdjectives = []
    for i in range(len(POScolumn) - 1):
        #		if (POScolumn[i]=='JJ'):
        #			print("adjNoun({})={} -- {}".format(i,POScolumn[i],POScolumn[i+1]))
        if POScolumn[i] == 'JJ' and POScolumn[i + 1].startswith('NN'):
            currentAdjIndex = i
            currentNounIndex = i + 1
            #			print("Creating candidate...")
            while currentNounIndex < len(POScolumn) and POScolumn[currentNounIndex].startswith('NN'):
                currentNounIndex += 1
            #			print("Creating candidate for {} -- {}".format(i,currentNounIndex))
            while currentAdjIndex >= 0 and POScolumn[currentAdjIndex] == 'JJ':
                newCandidate = Candidate(annotatedText, currentAdjIndex, (currentAdjIndex, currentAdjIndex),
                                         currentNounIndex - 1, (i + 1, currentNounIndex - 1))
                candidates.addCandidate(newCandidate)
                #				print("New Candidate {}".format(newCandidate))
                currentAdjIndex -= 1
    #	print(candidates)
    return candidates


def nounNounFinder(annotatedText):
    candidates = CandidateGroup()
    POScolumn = annotatedText.getColumn("POS")
    wordColumn = annotatedText.getColumn("word")
    pattern = ['of', 'is', 'was', 'were', 'am', 'had', 'will', 'are', 'have']
    # pattern = ['of']
    ignore_in_pattern = ['a', 'an', 'the']
    ignore_in_pattern.extend(pattern)
    for i in range(len(POScolumn) - 1):
        if POScolumn[i].startswith('NN') and (wordColumn[i + 1] in pattern):
            sourceNounIndex = i
            targetNounIndex = i + 1
            candidateFound = False
            condition1 = lambda x: (x < len(POScolumn)) and POScolumn[x].startswith('NN')
            condition2 = lambda x: (x < len(wordColumn)) and wordColumn[x] in ignore_in_pattern
            condition3 = lambda x: (x < len(POScolumn)) and POScolumn[x] == 'JJ'
            condition = lambda x: condition1(x) or condition2(x) or condition3(x)
            while condition(targetNounIndex):
                if condition(targetNounIndex) and not condition(targetNounIndex + 1):
                    if condition1(targetNounIndex):
                        candidateFound = True
                        break
                targetNounIndex += 1

            if candidateFound:
                if 'of' in wordColumn[sourceNounIndex:targetNounIndex+1]:
                    newCandidate = Candidate(annotatedText, sourceNounIndex, (sourceNounIndex, sourceNounIndex),
                                         targetNounIndex, (targetNounIndex, targetNounIndex))
                else:
                    newCandidate = Candidate(annotatedText, targetNounIndex, (targetNounIndex, targetNounIndex),
                                             sourceNounIndex, (sourceNounIndex, sourceNounIndex))
                candidates.addCandidate(newCandidate)

    return candidates


# Finds the verb and the next noun in the sentence
def verbNounFinder(annotatedText):
    candidates = CandidateGroup()
    POScolumn = annotatedText.getColumn("POS")
    wordColumn = annotatedText.getColumn("word")
    candidate = []
    currentAdjectives = []
    for i in range(len(POScolumn) - 1):
        if POScolumn[i].startswith('VB'):
            currentVerbIndex = i
            currentNounIndex = i
            while (currentNounIndex < len(POScolumn) and wordColumn[currentNounIndex] != "." and not (
                    POScolumn[currentNounIndex].startswith('NN'))):
                currentNounIndex += 1
            if currentNounIndex < len(POScolumn) and POScolumn[currentNounIndex].startswith('NN'):
                newCandidate = Candidate(annotatedText, currentVerbIndex, (currentVerbIndex, currentVerbIndex),
                                         currentNounIndex, (currentNounIndex, currentNounIndex))
                candidates.addCandidate(newCandidate)

    return candidates


# Need modification for sources or targets that are more than 1 word long
def candidateFromPair(annotatedText, source, target):
    if source in annotatedText.words:
        sourceIndex = annotatedText.words.index(source)
    else:
        sourceIndex = annotatedText.getColumn("lemma").index(source)

    if target in annotatedText.words:
        targetIndex = annotatedText.words.index(target)
    else:
        targetIndex = annotatedText.getColumn("lemma").index(target)

    sourceSpan = (sourceIndex, sourceIndex)
    targetSpan = (targetIndex, targetIndex)

    return Candidate(annotatedText, sourceIndex, sourceSpan, targetIndex, targetSpan)


# TODO: Write function that finds a verb and its object using a dependancy parser
'''
def verbObjFinder(annotatedText):
	candidates = CandidateGroup()
	text = annotatedText.getText()

	parser = StanfordDependencyParser()
	lemmatizer = WordNetLemmatizer()
	dependency_tree = [list(line.triples()) for line in parser.raw_parse(text)]
	dependencies = dependency_tree[0]
	verbLemma = ""
	obj = ""
	currentIndex = 0
	for dep in dependencies:
		if annotatedText.getElement(currentIndex, "word") in [',', ';', '-', '.', '?', '!']:
			currentIndex += 1

		if  "VB" in dep[0][1]:
			verbLemma = lemmatizer.lemmatize(dep[0][0], wordnet.VERB)
			verbIndex = currentIndex
			if ("obj" in dep[1] or "nsubjpass" in dep[1]):
				obj = dep[2][0]
				# NEED TO BE ABLE TO CREATE CANDIDATE FROM WORDS INSTEAD OF INDEXES
				#newCandidate = Candidate(annotatedText, objIndex, (objIndex, objIndex), )
		currentIndex += 1
'''


def testLabelFunction(candidates):
    results = MetaphorGroup()
    for c in candidates:
        if (c.getSource()[0] == c.getTarget()[0]):
            results.addMetaphor(Metaphor(c, True, 0.5))
        else:
            results.addMetaphor(Metaphor(c, False, 0.5))

    return results
