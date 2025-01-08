from fastapi import FastAPI
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



app = FastAPI()
app.title = "Aplicacion con FastAPI"

# Cargar el archivo CSV con las películas
movies_df = pd.read_csv("movies_transform.csv")

# Función auxiliar para convertir mes en español a número
def month_to_number(mes):
    """
    Esta función toma un mes en español y lo convierte a su número correspondiente.
    Por ejemplo, 'enero' será convertido a 1, 'febrero' a 2, etc.
    """
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    return meses.get(mes.lower())

# Función auxiliar para convertir día en español a nombre en inglés
def day_to_english(dia):
    """
    Esta función convierte un día de la semana en español a su nombre en inglés.
    Por ejemplo, 'lunes' será convertido a 'Monday', 'martes' a 'Tuesday', etc.
    """
    dias = {
        'lunes': 'Monday', 'martes': 'Tuesday', 'miercoles': 'Wednesday',
        'jueves': 'Thursday', 'viernes': 'Friday', 'sabado': 'Saturday', 'domingo': 'Sunday'
    }
    return dias.get(dia.lower())

# Función que devuelve la cantidad de películas estrenadas en un mes dado
@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    """
    Esta función recibe un mes en español y devuelve la cantidad de películas que fueron estrenadas en ese mes
    en la totalidad del dataset. 
    Ejemplo de retorno: "5 películas fueron estrenadas en el mes de enero".
    """
    mes_num = month_to_number(mes)
    if mes_num:
        count = movies_df[pd.to_datetime(movies_df['release_date'], errors='coerce').dt.month == mes_num].shape[0]
        return {"message": f"{count} películas fueron estrenadas en el mes de {mes}"}
    return {"error": "Mes no válido"}

# Función que devuelve la cantidad de películas estrenadas en un día dado
@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    """
    Esta función recibe un día en español y devuelve la cantidad de películas que fueron estrenadas en ese día
    en la totalidad del dataset.
    Ejemplo de retorno: "5 películas fueron estrenadas en los días lunes".
    """
    dia_ingles = day_to_english(dia)
    if dia_ingles:
        count = movies_df[pd.to_datetime(movies_df['release_date'], errors='coerce').dt.day_name() == dia_ingles].shape[0]
        return {"message": f"{count} cantidad de películas fueron estrenadas en los días {dia}"}
    return {"error": "Día no válido"}

# Función que devuelve el score de una película dada su título
@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    """
    Esta función recibe el título de una película y devuelve el título, el año de estreno y el score/popularidad.
    Ejemplo de retorno: "La película Titanic fue estrenada en el año 1997 con un score/popularidad de 85.5".
    """
    movie = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    if not movie.empty:
        title = movie.iloc[0]['title']
        year = movie.iloc[0]['release_date'][:4]
        score = movie.iloc[0]['popularity']
        return {"message": f"La película {title} fue estrenada en el año {year} con un score/popularidad de {round(score, 2)}"}
    return {"error": "Película no encontrada"}

# Función que devuelve el número de votos y el promedio de votos de una película
@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    """
    Esta función recibe el título de una película y devuelve el número de votos y el promedio de votos.
    Si la película tiene menos de 2000 votos, se devolverá un mensaje indicando que no cumple con la condición.
    Ejemplo de retorno: "La película Titanic fue estrenada en el año 1997. La misma cuenta con un total de 2500 valoraciones, con un promedio de 7.8".
    """
    movie = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    if not movie.empty:
        title = movie.iloc[0]['title']
        votes = movie.iloc[0]['vote_count']
        year = movie.iloc[0]['release_date'][:4]
        average_vote = movie.iloc[0]['vote_average']
        if votes >= 2000:
            return {"message": f"La película {title} fue estrenada en el año {year}, tiene un total de {votes} valoraciones, con un promedio de {round(average_vote, 2)}"}
        else:
            return {"message": f"La película {title} no cumple con la condición de tener al menos 2000 valoraciones"}
    return {"error": "Película no encontrada"}

# Función que devuelve la información de un actor (películas, retorno total y promedio)
@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    """
    Esta función recibe el nombre de un actor y devuelve el éxito del actor medido a través del retorno,
    la cantidad de películas en las que ha participado y el promedio de retorno por película.
    Ejemplo de retorno: "El actor Leonardo DiCaprio ha participado en 20 películas, con un retorno total de 500 millones y un promedio de 25 millones por película".
    """
    actor_movies = movies_df[movies_df['cast'].str.contains(nombre_actor, case=False, na=False)]
    if not actor_movies.empty:
        total_return = actor_movies['return'].sum()
        count_movies = actor_movies.shape[0]
        average_return = total_return / count_movies if count_movies > 0 else 0
        return {
            "message": f"El actor {nombre_actor} ha participado en {count_movies} películas, con un retorno total de {round(total_return, 2)} y un promedio de {round(average_return, 2)} por película"
        }
    return {"error": "Actor no encontrado en el dataset"}

# Función que devuelve la información de un director y las películas que ha dirigido
@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    """
    Esta función recibe el nombre de un director y devuelve el éxito del director medido a través del retorno,
    y una lista con el título, fecha de estreno, retorno, presupuesto y ganancia de las películas dirigidas.
    Ejemplo de retorno: "El director Christopher Nolan ha dirigido 10 películas con un retorno total de 1000 millones. Las películas son: ...".
    """
    director_movies = movies_df[movies_df['name_director'].str.contains(nombre_director, case=False, na=False)]
    
    if not director_movies.empty:
        retorno_dir = director_movies['return'].sum()
        movies_data = []
        for _, row in director_movies.iterrows():
            movies_data.append({
                "Titulo": row['title'],
                "Fecha": row['release_date'],
                "Retorno": row['return'],
                "Presupuesto": row['budget'],
                "Ganancia": row['revenue']
            })
        
        return {
            "Director": nombre_director,
            "Retorno": retorno_dir,
            "Movies": movies_data
        }
    
    return {"error": "Director no encontrado en el dataset"}

# Función para recomendar películas similares a una dada

# Vectorización con TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['title'])

# Calcular la similitud del coseno
cosine_similarities = cosine_similarity(tfidf_matrix)



@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    """
    Esta función recibe un título de película y devuelve una lista de las 5 películas más similares basadas
    en la similitud del coseno entre los títulos.
    Ejemplo de retorno: "Las películas más similares a Titanic son: ...]".
    """
    # Verificar si el título está en el DataFrame
    if titulo not in movies_df['title'].values:
        return {"error": f"No se encontró ninguna película con el título '{titulo}'."}
    
    # Establecer el índice de las películas por su título
    idx = movies_df[movies_df['title'] == titulo].index[0]
    
    # Obtener las similitudes de coseno de la película con todos los demás
    cosine_similarities = cosine_similarity(tfidf_matrix)
    
    # Obtener las similitudes con la película seleccionada
    sim_scores = list(enumerate(cosine_similarities[idx]))

    # Asegurarse de que cada puntaje sea un valor escalar
    sim_scores = [(i, score) for i, score in sim_scores if isinstance(score, (int, float))]

    # Ordenar las películas por puntaje de similitud en orden descendente
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los índices de las películas más similares (excluyendo la película dada)
    sim_scores = sim_scores[1:6]  # Obtener las 5 películas más similares

    # Obtener los títulos de las películas más similares
    movie_indices = [i[0] for i in sim_scores]
    recommended_movies = movies_df['title'].iloc[movie_indices]

    return recommended_movies.tolist()
