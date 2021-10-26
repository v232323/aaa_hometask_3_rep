class CountVectorizer():
    """
    Convert a collection of text documents to a matrix of token counts.
    """
    def __init__(self):
        self._features_names = []


    def fit_transform(self, corpus):
        """
        Learn the vocabulary dictionary and return document-term matrix.
        :param corpus: Some list of strings
        :return: Document-term matrix.
        """
        feature_names_set = set()
        for str in corpus:
            str_list = str.split(' ')
            for word in str_list:
                feature_names_set.add(word.lower())
        self._features_names = list(feature_names_set)

        result = []
        for str in corpus:
            str_list = str.lower().split(' ')
            matrix_row = [0]*len(list(feature_names_set))
            for word in str_list:
                for num, feature in enumerate(list(feature_names_set)):
                    if word == feature:
                        matrix_row[num] += 1
            result.append(matrix_row)
        return result


    def get_feature_names(self):
        """
        Array mapping from feature integer indices to feature name.
        :return: A list of feature names.
        """
        return self._features_names


if __name__ == '__main__':
    corpus = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste'
    ]

    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform(corpus)
    print(vectorizer.get_feature_names())
    print(count_matrix)

