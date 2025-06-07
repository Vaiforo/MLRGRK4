import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="Дашборд анализа данных", layout="wide")

st.title("Дашборд анализа данных и моделирования")
st.markdown("""
На этой странице представлены основные результаты анализа данных и оценки моделей машинного обучения для задачи прогнозирования срабатывания датчика дыма.
""")

image_dir = "figures"
def display_image_with_caption(image_path, caption):
    image = Image.open(os.path.join(image_dir, image_path))
    st.image(image, caption=caption, use_container_width=True)

col1, col2 = st.columns(2)

st.subheader("Aнализ данных:")
display_image_with_caption(
    "corr_matrix.png",
    "Тепловая карта корреляций\n"
)
st.write("Показывает взаимосвязь между признаками.")
st.write("Цвета отражают силу корреляции: от -1 до 1.")
st.write("Выявляет наиболее значимые признаки.")

with col1:
    st.subheader("Aнализ влияние важных признаков на цену:")
     # temperature.png
    display_image_with_caption(
        "temperature.png",
        "Зависимость Fire Alarm от Temperature[C]\n"
    )
    st.write("Помогает понять,что температура — один из ключевых факторов срабатывания датчика")
    
    # humidity.png
    display_image_with_caption(
        "humidity.png",
        "Зависимость Fire Alarm от Humidity[%]\n"
    )
    st.write("Показывает, что влажность также является важными параметром срабатывания дачика, но может иметь выбросы:")
    
    # pairplot.png
    display_image_with_caption(
        "pairplot.png",
        "Взаимосвязь важных признаков на целевую переменную.\n"
    )
    st.write("Видна положительная зависимость (чем выше температура, тем вероятнее сработает датчик), но разброс достаточно велик")
    st.write("По вертикальным полосам видно, что не сработавшие показания влажности нахотядся в одной небольшой области")
    st.write("По TVOC и ECo2 видно, что датчик срабатывает, при сильном превышении показателей этих параметров")
    
with col2:
    st.subheader("Oценки моделей машинного обучения (на предобработанных данных):")
    # knn_metrics_table.png
    display_image_with_caption(
        "knn_metrics_table.png",
        "Таблица метрик для KNN. \n"
    )

    # boosting_metrics_table.png
    display_image_with_caption(
        "boosting_metrics_table.png",
        "Таблица метрик для Gradient Boosting Classifier. \n"
    )

    # cat_metrics_table.png
    display_image_with_caption(
        "cat_metrics_table.png",
        "Таблица метрик для CatBoostClassifier. \n"
    )
    
    # bbagging_metrics_table.png
    display_image_with_caption(
        "bagging_metrics_table.png",
        "Таблица метрик для Bagging Classifier. \n"
    )

    # stacking_metrics_table.png
    display_image_with_caption(
        "stacking_metrics_table.png",
        "Таблица метрик для StackingClassifier. \n"
    )

    # mlp_metrics_table.png
    display_image_with_caption(
        "mlp_metrics_table.png",
        "Таблица метрик для MLPClassifier. \n"
    )
    st.write("Accuracy: Доля правильных предсказаний модели.")
    st.write("Precision: Доля правильных положительных предсказаний среди всех положительных предсказаний.")
    st.write("Recall: Доля правильно предсказанных положительных случаев среди всех реальных положительных случаев.")
    st.write("F1-score: Гармоническое среднее precision и recall, баланс между ними.")
    st.write("Support: Количество образцов для каждого класса.")
