# Metaphor Project

# User Guide

The file main.py represents the process of finding metaphors in a text:
1. Text Segmentation
2. Finding candidates for the metaphors: a candidate is a pair of word: adjective-noun or verb-noun 
3. Labeling the metaphors: Is a candidate metaphorical or literal?

## Importing a text

The framework can find metaphors in one or several texts.
You can import the text from a string written in the command line, from a text file, or from a CSV/Excel file which can contain multiple texts on different rows.
Each text will be processed independently. The name of the column should be one of the following:
* text
* Text
* sentence
* Sentence

## Execution: The Command Line
Arguments:
* -ml or --labeler followed by a string between _"_ containing one or several metaphor labeler IDs separated by spaces.
* -cf or --finder followed by one of the following string. If absent, the default value is adjNoun. 
    * adjNoun
    * verbNoun
* -v or --verbose. The default value is False.
    * Print the different steps of the process
* -f or --file followed by a string between _"_ representing the path to the file containing the text to analyze.
* -s or --string followed by a string between _"_ representing the text we want to analyze.
* -cg or --cgenerator:
    * Useful when combined with an excel or csv file. Use word pair in the file as candidates instead of looking for candidates in the annotated text
    * Default value: False
* -csv followed by a string between _"_ representing the path of the file containing the exported results.

If no string or text file is specified in the command line then a default text is used.

##How to Add a New Metaphor-Labeling Function?
Your function must be defined in a new file placed in the _modules_ folder.
You must add it to the 

### Input
The input of the function must be:
* candidates
    * Type: Object of class _CandidateGroup_
* cand_type:
    * Type: string
    * Value: _"adjNoun"_ or _"verbNoun"_
    * Usage: Corresponds to a database
* verbose:
    * Type: Boolean
    * Usage: Display some information if its value is _True_
    
### Output
The output of the function must be an object of class _MetaphorGroup_

# Documentation 

## Data structures

### CandidateGroup
* Variables
    * candidates: list of objects of class Candidate
    * size: number of elements in the list above
* Methods
    * addCandidate(candidate): Add the element candidate to the list candidates and increment the variable size by 1
    * getCandidate(index): Return the candidate of index index in the list candidates
    * \_\_iter\_\_()
    * \_\_str\_\_()
    
### MetaphorGroup
* Variables
    * metaphors: list of objects of class Metaphor
    * size: number of elements in the list above
* Methods
    * addMetaphor(metaphor): Add the element metaphor to the list metaphors and increment the variable size by 1
    * getMetaphor(index): Return the metaphor of index index in the list metaphors
    * writeToCSV()
    * \_\_iter\_\_()
    * \_\_str\_\_()
    
### Candidate
* Variables
    * annotatedText: object of class AnnotatedText
    * sourceIndex: index of the source in the annotatedText
    * sourceSpan: 2-tuple = (index of the first word in the source, index of the last word in the source)
    * targetIndex: index of the target in the annotatedText
    * targetSpan: 2-tuple = (index of the first word in the  target , index of the last word in the  target) 
* Methods
    * getSource(): return the first word of the source
    * getTarget(): return the first word of the target
    * getFullSource()
    * getFullTarget()
    * \_\_stringAdder(): used in the getFull... functions
    
### Metaphor
* Variables
    * candidate: object of class candidate
    * result: boolean
    * confidence: number between 0 and 1
* Methods
    * getSource(): return candidate.getFullSource()
    * getTarget(): return candidate.getFullTarget()
    * getResult()
    * getConfidence()
    * \_\_str\_\_()

### The MetaphorIdentification Class

This is the core of the framework. It handles the metaphor labeling process
from start to finish.

Defined in _/new\_structure/modules/datastructs/MetaphorIdentification.py_.

It has 6 fields:
* mLabelers: a dictionary mapping IDs (string) to functions
* cFinders: a dictionary mapping IDs (string) to functions
* rawText: string
* annotatedText: class AnnotatedText from _modules/datastructs/annotated_text.py_
* candidates: class CandidateGroup from _modules/datastructs/candidate_group.py_
* metaphors: class MetaphorGroup from _modules/datastructs/labeled_metaphor_list.py_
