"""Predict labels using vectorizer and trained model."""
from pickle import load
from sys import argv
from scipy.sparse import hstack
from argparse import ArgumentParser


def read_lines_from_file(file_path):
    """Read lines from file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return [line.strip() for line in file_read.readlines() if line.strip()]


def read_text_from_file(file_path):
    """Read text from file."""
    with open(file_path, 'r', encoding='utf-8') as file_read:
        return file_read.read().strip().replace('\n', ' ')


def write_list_to_file(file_path, data_list):
    """Write list to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write('\n'.join(data_list))


def write_text_to_file(file_path, data):
    """Write text to a file."""
    with open(file_path, 'w', encoding='utf-8') as file_write:
        file_write.write(data)


def create_test_tfidf(test_data, tfidf_vect):
    """Create test tf idf for test data."""
    return tfidf_vect.transform(test_data)


def load_object_from_file(pickle_file):
    """Load an object from a file."""
    with open(pickle_file, 'rb') as file_load:
        return load(file_load)


def predict_on_features(classifier, features):
    """Predict output based on features."""
    return classifier.predict(features)


def main():
    """Pass arguments and call functions here."""
    parser = ArgumentParser("Program for predicting domain of text")
    parser.add_argument('--input', dest='inp', help='Enter the input file path.')
    parser.add_argument('--word', dest='word', help='Enter the word vectorizer file path.')
    parser.add_argument('--char', dest='char', help='Enter the character vectorizer file path.')
    parser.add_argument('--model', dest='model', help='Enter the saved model file path.')
    parser.add_argument('--output', dest='out', help='Enter the output file path.')
    args = parser.parse_args()
    test_file = args.inp
    word_tfidf_vect_file = args.word
    char_tfidf_vect_file = args.char
    classifier_file = args.model
    predicted_file = args.out
    word_tfidf_vect = load_object_from_file(word_tfidf_vect_file)
    char_tfidf_vect = load_object_from_file(char_tfidf_vect_file)
    test_lines = read_lines_from_file(test_file)
    test_data = test_lines
    word_test_tfidf = create_test_tfidf(test_data, word_tfidf_vect)
    char_test_tfidf = create_test_tfidf(test_data, char_tfidf_vect)
    combined_test_tfidf = hstack([word_test_tfidf, char_test_tfidf])
    classifier = load_object_from_file(classifier_file)
    predicted_labels = predict_on_features(classifier, combined_test_tfidf)
    write_list_to_file(predicted_file, predicted_labels)


if __name__ == '__main__':
    main()
