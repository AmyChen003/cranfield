import pandas as pd
from gensim.models import doc2vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer,TfidfVectorizer
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import gensim, logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from string import punctuation
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

'''Please feel free to modify this function (load_models)
   Make sure the function return the models required for the function below evaluate model
'''

stop_words = set(stopwords.words('english')+list(punctuation))
stemmer = PorterStemmer()


def load_models():
    vectorizer = joblib.load('resources/vectorizer.pkl')
    clf = joblib.load('resources/classifier.pkl')
    return [vectorizer, clf]


'''Please feel free to modify this function (evaluate_model)
  Make sure the function only take three parameters: 1) Model (one or more model files), 2) Query, and 3) Document.
  The function always should return one of the positions (classes) and the confidence. In my case confidence is always 0.5.
  Preferably implement the function in a different file and call from here. Make sure keep everything else the same.
'''


def evaluate_model(model, query, document):

    query_vec = model[0].transform([query['query']])
    title_vec = model[0].transform([document['title']])
    body_vec = model[0].transform([document['body']])
    comm_terms = common_terms(query['query'], document['body'])
    cos1 = cosine_similarity(query_vec, title_vec)
    cos2 = cosine_similarity(query_vec, body_vec)
    result = model[1].predict([[cos1,cos2,comm_terms ]])
    return result[0],0.5


def cosine_similarity(x,y):
    cos = cosine(x.toarray()[0], y.toarray()[0])
    if np.isfinite(cos):
        return cos
    return 0.0


def cosine_similarity_w2v(x,y):
    if x is None or y is None:
        return 0.0
    cos = cosine(np.array(x), np.array(y))
    if np.isfinite(cos):
        return cos
    return 0.0


def common_terms(x,y) :
    count =0
    second = y.split(" ")
    for each in x.split(" "):
        if each in second:
            count = count +1
    return count/len(second)


def convert_to_w2v(text, word2vec):
    tokens = text.split(" ")
    pattern_vector = np.zeros(word2vec.layer1_size)
    n_words = 0

    if len(tokens) > 1:
        for t in tokens:
            if t in word2vec:
                vector = word2vec[t.strip()]
                pattern_vector = np.add(pattern_vector,vector)
                n_words += 1
                pattern_vector = np.divide(pattern_vector,n_words)
    return pattern_vector.tolist()


def convert_to_d2v(text, doc2vec):
    tokens = text.split(" ")
    np.array(doc2vec.infer_vector(tokens))


def create_model(all_documents_file, relevance_file,query_file):
    '''Step 1. Creating  a dataframe with three fields query, title, and relevance(position)'''
    documents = pd.read_json(all_documents_file)[["id", "title", "body"]]

    query_file = pd.read_json(query_file)[["query number","query" ]]
    relevance = pd.read_json(relevance_file)[["query_num", "position", "id"]]
    relevance_with_values = relevance.merge(query_file,left_on ="query_num", right_on="query number")[ ["id","query", "position"]]\
        .merge(documents,left_on ="id", right_on="id") [["query", "position", "title", "body"]]

    relevance_with_values['common_terms'] = relevance_with_values.apply(lambda x: common_terms(x['title'], x['query']), axis =1)

    '''Step 2. Creating  a column for creating index'''

    relevance_with_values["all_text"] = relevance_with_values.apply( lambda x : x["query"] + x["title"] + x["body"] , axis =1)

    ''' Step 3. Creating a model for generating TF feature'''
    vectorizer = TfidfVectorizer()
    vectorizer.fit(relevance_with_values["all_text"])

    '''Step 3.1 Creating a model for word2vec'''

    relevance_with_values["all_text_tokens"] = relevance_with_values.apply(lambda x: x["all_text"].split(" "), axis=1)

    w2v_model = gensim.models.Word2Vec(relevance_with_values["all_text_tokens"], min_count=1,workers=4)

    w2v_model.save('resources/w2v_.model')

    '''Step 3.2 Creating a model for doc2vec'''
    all_sentences=[]
    sentences = []
    i = 0

    for index, row in relevance_with_values.iterrows():
         sentences.append(LabeledSentence(row["all_text"].split(), ['SENT_%s' % index]))
    for t in sentences:
        i+=1
        stem_filtered_sentence = []
        stop_filtered_sentence = []
        for item in t[0]:
            if item not in stop_words:
                stop_filtered_sentence.append(item)
        for s in stop_filtered_sentence:
            stem_filtered_sentence.append(stemmer.stem(s))
        all_sentences.append(LabeledSentence(stem_filtered_sentence, ['SENT_%s' % i]))

    d2v_model = gensim.models.Doc2Vec(all_sentences)

    d2v_model.save('resources/d2v_.model')

    ''' Step 4. Saving the model for TF features'''
    joblib.dump(vectorizer, 'resources/vectorizer.pkl')

    ''' Step 5. Converting query and title to vectors and finding cosine similarity of the vectors'''
    relevance_with_values["doc_vec"] = relevance_with_values.apply(lambda x: vectorizer.transform([x["title"]]), axis=1)

    relevance_with_values["doc_vec_w2v"] = relevance_with_values.apply(lambda x: convert_to_w2v(x["title"],w2v_model), axis=1)
    relevance_with_values["doc_vec_d2v"] = relevance_with_values.apply(lambda x: convert_to_d2v(x["title"], d2v_model),
                                                                       axis=1)

    relevance_with_values["query_vec_w2v"] = relevance_with_values.apply(
        lambda x: convert_to_w2v(x["query"], w2v_model), axis=1)
    relevance_with_values["query_vec_d2v"] = relevance_with_values.apply(
        lambda x: convert_to_d2v(x["query"], d2v_model), axis=1)
    relevance_with_values["body_vec_w2v"] = relevance_with_values.apply(
        lambda x: convert_to_w2v(x["body"], w2v_model), axis=1)
    relevance_with_values["body_vec_d2v"] = relevance_with_values.apply(
        lambda x: convert_to_d2v(x["body"], d2v_model), axis=1)

    relevance_with_values["body_vec"] = relevance_with_values.apply(lambda x: vectorizer.transform([x["body"]]), axis=1)
    relevance_with_values["query_vec"] = relevance_with_values.apply(lambda x: vectorizer.transform([x["query"]]), axis =1)
    relevance_with_values["cosine"]  = relevance_with_values.apply(lambda x: cosine_similarity(x['doc_vec'], x['query_vec']), axis=1)

    relevance_with_values["cosine_w2v"] = relevance_with_values.apply(
        lambda x: cosine_similarity_w2v(x['doc_vec_w2v'], x['query_vec_w2v']), axis=1)
    relevance_with_values["cosine_d2v"] = relevance_with_values.apply(
        lambda x: cosine_similarity_w2v(x['doc_vec_d2v'], x['query_vec_d2v']), axis=1)
    relevance_with_values["cosine_body_w2v"] = relevance_with_values.apply(
        lambda x: cosine_similarity_w2v(x['body_vec_w2v'], x['query_vec_w2v']), axis=1)
    relevance_with_values["cosine_body_d2v"] = relevance_with_values.apply(
        lambda x: cosine_similarity_w2v(x['body_vec_d2v'], x['query_vec_d2v']), axis=1)
    relevance_with_values["cosine_body"]  = relevance_with_values.apply(lambda x: cosine_similarity(x['body_vec'], x['query_vec']), axis=1)
    ''' Step 6. Defining the feature and label  for classification'''

    X = relevance_with_values[["cosine","cosine_body","cosine_d2v"]]
    Y = relevance_with_values["position"]

    ''' Step 7. Splitting the data for validation'''
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)

    ''' Step 8. Classification and validation'''
    target_names = ['1', '2', '3','4']

    clf = KNeighborsClassifier(n_neighbors=10,weights='uniform',algorithm='auto',leaf_size=60,p=5,
                               metric='minkowski',metric_params=None,n_jobs=1 ).fit(X_train, y_train)
    print classification_report(y_test, clf.predict(X_test), target_names=target_names)

    ''' Step 9. Saving the data '''
    joblib.dump(clf, 'resources/classifier.pkl')


if __name__ == '__main__':
    create_model("resources/cranfield_data.json", "resources/cranqrel.json", "resources/cran.qry.json")
