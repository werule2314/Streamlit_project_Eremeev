import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(layout="wide")
st.title('Рекомендательная система для фильмов с аналитическими возможностями')


# Генерация данных
def generate_data():
    np.random.seed(42)
    users = pd.DataFrame({
        'user_id': range(1, 101),
        'age': np.random.randint(18, 55, size=100),
        'gender': np.random.choice(['M', 'F'], size=100)
    })
    movies = pd.DataFrame({
        'movie_id': range(1, 101),
        'title': [f'Movie {i}' for i in range(1, 101)],
        'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Fantasy', 'Horror', 'Romance'], size=100)
    })
    ratings = pd.DataFrame({
        'user_id': np.random.randint(1, 101, size=500),
        'movie_id': np.random.randint(1, 101, size=500),
        'rating': np.random.randint(1, 6, size=500),
        'timestamp': pd.date_range(start='2022-01-01', periods=500)
    })
    return users, movies, ratings


users, movies, ratings = generate_data()


# Предобработка данных
def preprocess_data(ratings):
    ratings.dropna(inplace=True)
    return ratings


ratings = preprocess_data(ratings.copy())


# Визуализация данных
def visualize_data(movies, ratings):
    genre_counts = movies['genre'].value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(x='genre', data=movies, ax=ax[0])
    ax[0].set_title('Распределение жанров фильмов')
    ax[0].tick_params(axis='x', rotation=45)

    sns.histplot(ratings['rating'], kde=False, bins=5, ax=ax[1])
    ax[1].set_title('Распределение рейтингов фильмов')
    st.pyplot(fig)


visualize_data(movies, ratings)


# Кластеризация фильмов по рейтингу
def cluster_movies(ratings):
    scaler = StandardScaler()
    ratings_scaled = scaler.fit_transform(ratings[['rating']])
    kmeans = KMeans(n_clusters=3, random_state=42)
    ratings['cluster'] = kmeans.fit_predict(ratings_scaled)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x='movie_id', y='rating', hue='cluster', data=ratings, palette='viridis', ax=ax)
    ax.set_title('Кластеризация фильмов по рейтингу')
    st.pyplot(fig)


cluster_movies(ratings)


# Прогнозирование рейтинга
def predict_rating(model, movie_id):
    movie_data = np.array([[movie_id]])
    predicted_rating = model.predict(movie_data)[0]
    return predicted_rating


# Обучение модели для прогнозирования
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(ratings[['movie_id']], ratings['rating'])

# Импорт библиотек уже добавлен в начале кода

# Продолжение кода после обучения модели для прогнозирования...

# Добавляем боковую панель для навигации
st.sidebar.title("Навигация")
analysis_type = st.sidebar.radio(
    "Выберите раздел анализа",
    ('Главная', 'Визуализация данных', 'Кластеризация фильмов', 'Прогнозирование рейтинга', 'Выводы и интерпретации')
)

# На основе выбора показываем соответствующий раздел
if analysis_type == 'Главная':
    st.header('Добро пожаловать в систему рекомендаций фильмов!')
    st.markdown("...")

elif analysis_type == 'Визуализация данных':
    st.header('Визуализация данных')
    visualize_data(movies, ratings)

elif analysis_type == 'Кластеризация фильмов':
    st.header('Кластеризация фильмов по рейтингу')
    cluster_movies(ratings)

elif analysis_type == 'Прогнозирование рейтинга':
    st.header('Прогнозирование рейтинга фильма')
    # Весь блок кода для прогнозирования рейтинга...
    movie_title = st.selectbox('Выберите фильм', movies['title'])
    selected_movie_id = movies[movies['title'] == movie_title].iloc[0]['movie_id']

    if st.button('Прогнозировать рейтинг для выбранного фильма'):
        predicted_rating = predict_rating(model, selected_movie_id)
        st.write(f'Прогнозируемый рейтинг для фильма "{movie_title}": {predicted_rating:.2f}')

elif analysis_type == 'Выводы и интерпретации':
    st.header('Выводы и интерпретации')
    st.markdown("""
        - **Распределение жанров**: График показывает, какие жанры фильмов наиболее популярны среди выбранной выборки.
        - **Распределение рейтингов**: Большинство фильмов имеют средний рейтинг около 3-4 баллов. Это указывает на умеренное удовлетворение качеством фильмов у зрителей.
        - **Кластеризация рейтингов**: Помогает определить группы фильмов с аналогичными рейтингами и может указывать на схожие предпочтения у групп зрителей.
        """)
else:
    st.error("Выбран неверный раздел.")
