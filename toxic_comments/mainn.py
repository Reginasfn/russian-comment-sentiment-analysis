import streamlit as st
import mysql.connector
import pandas as pd
from model_loader import predict_toxicity

# общие настройки
st.set_page_config(
    page_title="Анализ тональности комментариев",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Анализ комментариев")

# справочная информация
with st.sidebar:
    st.header("Справка")

    st.write("""
        Приложение анализирующее комментарии пользователей.

        Модель машинного обучения определяет тональность комментария.

        Возможные классы:

        • нормальный  
        • оскорбление  
        • угроза  
        • непристойность  

        На второй вкладке обработка загруженного
        датасета в базу данных и его статистика.
        """)

# подключение к бдшке
def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="27022006",
        database="tone_comment"
    )


# сохранение нового комментария
def save_comment(comment, probabilities, prediction):

    conn = get_connection()
    cursor = conn.cursor()

    sql = """
    INSERT INTO comments
    (comment, prob_normal, prob_insult, prob_threat, prob_obscene, prediction)
    VALUES (%s,%s,%s,%s,%s,%s)
    """

    values = (
        comment,
        float(probabilities[0]),
        float(probabilities[1]),
        float(probabilities[2]),
        float(probabilities[3]),
        prediction
    )

    cursor.execute(sql, values)
    conn.commit()

    cursor.close()
    conn.close()


# загрузка коммов
def load_comments():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM comments", conn)
    conn.close()
    return df


# обновление датасета
def update_dataset():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, comment FROM comments")
    rows = cursor.fetchall()

    progress = st.progress(0)

    total = len(rows)

    for i, row in enumerate(rows):

        comment_id = row[0]
        text = row[1]

        result = predict_toxicity(text)

        probs = result["probabilities"]
        pred = result["predicted_class"]

        update_sql = """
        UPDATE comments
        SET prob_normal=%s,
            prob_insult=%s,
            prob_threat=%s,
            prob_obscene=%s,
            prediction=%s
        WHERE id=%s
        """

        values = (
            float(probs[0]),
            float(probs[1]),
            float(probs[2]),
            float(probs[3]),
            pred,
            comment_id
        )

        cursor.execute(update_sql, values)

        progress.progress((i + 1) / total)

    conn.commit()
    cursor.close()
    conn.close()

# страницы-вкладки
tab1, tab2 = st.tabs(["Новый комментарий", "Анализ датасета"])

# новый комментарий 1 страничка
with tab1:
    st.header("Анализ нового комментария")
    comment = st.text_area("Введите комментарий", height=150)
    if st.button("Анализировать комментарий"):

        if comment.strip() == "":
            st.warning("Введите текст комментария.")
        else:

            result = predict_toxicity(comment)

            probabilities = result["probabilities"]
            predicted_class = result["predicted_class"]

            st.success(f"Предсказанный класс: **{predicted_class}**")

            st.write("### Вероятности классов")
            st.write(f"Нормальный: {probabilities[0]:.3f}")
            st.write(f"Оскорбление: {probabilities[1]:.3f}")
            st.write(f"Угроза: {probabilities[2]:.3f}")
            st.write(f"Непристойность: {probabilities[3]:.3f}")

            chart_data = {
                "Нормальный": probabilities[0],
                "Оскорбление": probabilities[1],
                "Угроза": probabilities[2],
                "Непристойность": probabilities[3]
            }
            st.bar_chart(chart_data)
            save_comment(comment, probabilities, predicted_class)
            st.info("Комментарий сохранён в базе данных.")

# анализ датасета 2 страничка
with tab2:
    st.header("Анализ всего датасета")

    if st.button("Обработать весь датасет"):
        st.write("Обработка комментариев...")
        update_dataset()
        st.success("Все комментарии обновлены!")
    df = load_comments()

    # метрики
    st.subheader("Общая статистика")
    col1, col2, col3 = st.columns(3)
    col1.metric("Всего комментариев", len(df))
    col2.metric("Нормальные", len(df[df["prediction"] == "нормальный"]))
    col3.metric("Токсичные", len(df[df["prediction"] != "нормальный"]))

    # статистика классов
    st.subheader("Статистика классов")
    stats = df["prediction"].value_counts()
    st.bar_chart(stats)
    st.write("Количество комментариев по классам:")
    st.write(stats)

    # просмотр датасета
    st.subheader("Просмотр данных")
    st.dataframe(df.head(100))
    
    # комментарии по классам
    st.subheader("Примеры комментариев по классам")
    classes = ["нормальный", "оскорбление", "угроза", "непристойность"]

    for class_name in classes:
        st.markdown(f"### Класс: {class_name}")
        class_df = df[df["prediction"] == class_name]

        if len(class_df) == 0:
            st.write("Нет комментариев данного класса.")
            continue

        if st.button(f"Показать другие 5 ({class_name})"):
            sample = class_df.sample(n=min(5, len(class_df)))
        else:
            sample = class_df.sample(n=min(5, len(class_df)))

        st.table(sample[["comment"]])