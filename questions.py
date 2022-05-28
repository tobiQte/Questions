import nltk
import sys
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    corpus = {}
    filepath = ""

    # read through directory
    for files in os.listdir(directory):

        # join file path plus file name for full file location
        filepath = os.path.join(directory, str(files))

        # loop through files and read corpus into key
        with open(filepath, "r") as file:
            corpus[files] = file.read()

    return corpus


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    documents = []
    #tokenize and lowercase document
    tokenized_document = word_tokenize(document.lower())

    #loop through words and remove punctuation and stopwords

    for word in tokenized_document:
        if not all(char  in string.punctuation for char in word):
            if word not in stopwords.words("english"):
                documents.append(word)

    return documents


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """

    # Get all words in corpus
    words = set()
    for filename in documents:
        words.update(documents[filename])

    # Calculate IDFs
    idfs = dict()
    for word in words:
        f = sum(word in documents[filename] for filename in documents)
        idf = math.log(len(documents) / f)
        idfs[word] = idf

    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """

    tfidf = {}
    for file in files:
        tfidf[file] = 0
        for word in query:
            tfidf[file] += idfs[word] * files[file].count(word)

    return [key for key, value in sorted(tfidf.items(), key=lambda item: item[1], reverse=True)][:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    idf_sentence = {}
    for sentence in sentences:
        query_term_density = 0
        idf = 0
        idf_sentence[sentence] = 0
        for word in query:
            if word in sentences[sentence]:
                idf += idfs[word]
            query_term_density +=  sentences[sentence].count(word) / len(sentences[sentence])
        idf_sentence[sentence] = (idf, query_term_density)

    return [key for key, value  in sorted(idf_sentence.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)][:n]


if __name__ == "__main__":
    main()
