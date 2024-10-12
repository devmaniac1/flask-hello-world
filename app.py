from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

# Combine article title, description, and content for TF-IDF vectorization
def get_article_corpus(articles):
    return [article['title'] + ' ' + article['description'] + ' ' + article['content'] for article in articles]

# Function to recommend articles based on user preferences and time spent on each category
def recommend_articles(user_preferences, articles):
    vectorizer = TfidfVectorizer()
    corpus = get_article_corpus(articles)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    scores = []
    for idx, article in enumerate(articles):
        article_vector = vectorizer.transform([corpus[idx]])

        article_score = 0
        for tag in article['tags']:
            if tag in user_preferences:
                time_spent = user_preferences[tag]
                article_score += time_spent * 1.5  # Boost score based on time spent on that preference

        scores.append(article_score)

    ranked_articles = [articles[i] for i in np.argsort(scores)[::-1]]
    return ranked_articles

# Endpoint to get recommended articles
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_preferences = data.get('user_preferences')
    articles = data.get('articles')

    if not user_preferences or not articles:
        return jsonify({'error': 'User preferences and articles are required'}), 400
    
    recommended_articles = recommend_articles(user_preferences, articles)
    
    return jsonify(recommended_articles), 200

@app.route('/', methods=['GET'])
def checkStatus():
    return jsonify({
        'status': 'Running',
        'message': 'Flask backend is up and running!',
        'version': '1.0.0'
    }), 200

if __name__ == '__main__':
    app.run(port=3000,debug=True)
