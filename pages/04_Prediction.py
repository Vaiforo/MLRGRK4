# import streamlit as st
# import pandas as pd
# import pickle
# import os
# from sklearn.preprocessing import LabelEncoder

# st.set_page_config(page_title="Предсказание", layout="centered")
# st.title("Прогнозирование срабатывания датчика дыма")

# st.markdown(
#     "Загрузите CSV-файл или введите данные вручную для получения прогноза.")

# models_dir = "models"

# if not os.path.exists(models_dir):
#     st.error(f"Папка с моделями не найдена: {models_dir}")
# else:
#     model_files = [f for f in os.listdir(
#         models_dir) if f.endswith('.pkl') or f.endswith('.json')]

#     if not model_files:
#         st.warning("В папке models нет моделей (.pkl)")
#     else:
#         selected_model = st.selectbox("Выберите модель", model_files)

#         model_path = os.path.join(models_dir, selected_model)
#         if selected_model.endswith('.pkl'):
#             with open(model_path, "rb") as f:
#                 model = pickle.load(f)
#             st.success(f"Модель '{selected_model}' (.pkl) загружена")
#         else:
#             st.error("Неподдерживаемый формат модели!")
#             model = None

#         if model is not None:
#             model_columns = model.feature_names_in_ if hasattr(
#                 model, 'feature_names_in_') else model.get_booster().get_fscore().keys()

#             col1, col2 = st.columns(2)

#             with col1:
#                 st.subheader("Загрузите CSV-файл")
#                 uploaded_file = st.file_uploader(
#                     "Выберите CSV-файл", type=["csv"])
#                 if uploaded_file:
#                     try:
#                         df_uploaded = pd.read_csv(uploaded_file)
#                         st.success("Файл успешно загружен!")
#                         st.write("Предпросмотр данных:")
#                         st.dataframe(df_uploaded.head())

#                         df_uploaded = df_uploaded.fillna(0)

#                         df_uploaded = pd.get_dummies(
#                             df_uploaded, drop_first=True)

#                         missing_columns = set(
#                             model_columns) - set(df_uploaded.columns)
#                         for col in missing_columns:
#                             df_uploaded[col] = 0

#                         df_uploaded = df_uploaded[model_columns]

#                         X = df_uploaded
#                         predictions = model.predict(X)
#                         df_uploaded['predicted'] = predictions
#                         st.download_button(
#                             label="Скачать с предсказаниями",
#                             data=df_uploaded.to_csv(index=False),
#                             file_name="predictions.csv",
#                             mime="text/csv"
#                         )

#                         st.subheader("Предсказания для загруженных данных:")
#                         st.write(df_uploaded.head(10))

#                     except Exception as e:
#                         st.error(f"Ошибка при обработке файла: {e}")

#             with col2:
#                 st.subheader("Ввод параметров объекта недвижимости вручную")

#                 temp = st.number_input(
#                     "Температура [C]:", min_value=-50.0, max_value=100.0, value=20.0)
#                 hum = st.number_input(
#                     "Влажность [%]:", min_value=0.0, max_value=100.0, value=50.0)
#                 tvoc = st.number_input(
#                     "TVOC [ppb]:", min_value=0.0, max_value=5000.0, value=100.0)
#                 eco2 = st.number_input(
#                     "eCO2 [ppm]:", min_value=400.0, max_value=5000.0, value=400.0)
#                 h2 = st.number_input(
#                     "Raw H2:", min_value=10000.0, max_value=20000.0, value=13000.0)
#                 eth = st.number_input(
#                     "Raw Ethanol:", min_value=15000.0, max_value=25000.0, value=20000.0)
#                 hpa = st.number_input(
#                     "Давление [hPa]:", min_value=900.0, max_value=1100.0, value=939.0)
#                 pm = st.number_input(
#                     "PM1.0:", min_value=0.0, max_value=10.0, value=1.0)

#                 input_data = pd.DataFrame({
#                     'Temperature[C]': [temp],
#                     'Humidity[%]': [hum],
#                     'TVOC[ppb]': [tvoc],
#                     'eCO2[ppm]': [eco2],
#                     'Raw H2': [h2],
#                     'Raw Ethanol': [eth],
#                     'Pressure[hPa]': [hpa],
#                     'PM1.0': [pm],
#                 })

#                 missing_columns = set(model_columns) - set(input_data.columns)
#                 for col in missing_columns:
#                     input_data[col] = 0

#                 input_data = input_data[model_columns]

#                 if st.button("Получить прогноз срабатывания датчика"):
#                     prediction = model.predict(input_data)[0]

#                     formatted_prediction = "{:,.0f}".format(
#                         prediction).replace(",", " ")

#                     st.success(
#                         f"Датчик сработа: **{'да' if formatted_prediction else 'нет'}**")
import streamlit as st
import pandas as pd
import pickle
import os
import catboost as xgb
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Предсказание", layout="centered")
st.title("Прогнозирование срабатывания датчика дыма")

st.markdown(
    "Загрузите CSV-файл или введите данные вручную для получения прогноза.")

models_dir = "models"

if not os.path.exists(models_dir):
    st.error(f"Папка с моделями не найдена: {models_dir}")
else:
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]

    if not model_files:
        st.warning("В папке models нет моделей (.pkl)")
    else:
        selected_model = st.selectbox("Выберите модель", model_files)
        model_path = os.path.join(models_dir, selected_model)

        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)

                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                    model_columns = model_data.get('feature_names', None)
                else:
                    model = model_data
                    model_columns = None
            st.success(f"Модель '{selected_model}' (.pkl) загружена")
        except Exception as e:
            st.error(f"Ошибка при загрузке модели '{selected_model}': {e}")
            model = None
            model_columns = None

        if model is not None:
            if model_columns is None:
                if hasattr(model, 'feature_names_in_'):
                    model_columns = model.feature_names_in_
                elif hasattr(model, 'get_booster'):
                    model_columns = model.get_booster().get_fscore().keys()
                else:
                    model_columns = [
                        'Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]',
                        'Raw H2', 'Raw Ethanol', 'Pressure[hPa]', 'PM1.0'
                    ]
                    st.warning(
                        "Имена признаков не найдены в модели. Используется запасной список признаков.")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Загрузите CSV-файл")
                uploaded_file = st.file_uploader(
                    "Выберите CSV-файл", type=["csv"])
                if uploaded_file:
                    try:
                        df_uploaded = pd.read_csv(uploaded_file)
                        st.success("Файл успешно загружен!")
                        st.write("Предпросмотр данных:")
                        st.dataframe(df_uploaded.head())

                        df_uploaded = df_uploaded.fillna(0)

                        df_uploaded = pd.get_dummies(
                            df_uploaded, drop_first=True)

                        missing_columns = set(
                            model_columns) - set(df_uploaded.columns)
                        for col in missing_columns:
                            df_uploaded[col] = 0

                        df_uploaded = df_uploaded[model_columns]

                        X = df_uploaded
                        predictions = model.predict(X)
                        df_uploaded['predicted'] = predictions

                        st.download_button(
                            label="Скачать с предсказаниями",
                            data=df_uploaded.to_csv(index=False),
                            file_name="predictions.csv",
                            mime="text/csv"
                        )

                        st.subheader("Предсказания для загруженных данных:")
                        st.write(df_uploaded.head(10))

                    except Exception as e:
                        st.error(f"Ошибка при обработке файла: {e}")

            with col2:
                st.subheader("Ввод параметров объекта недвижимости вручную")

                temp = st.number_input(
                    "Температура [C]:", min_value=-50.0, max_value=100.0, value=20.0)
                hum = st.number_input(
                    "Влажность [%]:", min_value=0.0, max_value=100.0, value=50.0)
                tvoc = st.number_input(
                    "TVOC [ppb]:", min_value=0.0, max_value=5000.0, value=100.0)
                eco2 = st.number_input(
                    "eCO2 [ppm]:", min_value=400.0, max_value=5000.0, value=400.0)
                h2 = st.number_input(
                    "Raw H2:", min_value=10000.0, max_value=20000.0, value=13000.0)
                eth = st.number_input(
                    "Raw Ethanol:", min_value=15000.0, max_value=25000.0, value=20000.0)
                hpa = st.number_input(
                    "Давление [hPa]:", min_value=900.0, max_value=1100.0, value=939.0)
                pm = st.number_input(
                    "PM1.0:", min_value=0.0, max_value=10.0, value=1.0)

                input_data = pd.DataFrame({
                    'Temperature[C]': [temp],
                    'Humidity[%]': [hum],
                    'TVOC[ppb]': [tvoc],
                    'eCO2[ppm]': [eco2],
                    'Raw H2': [h2],
                    'Raw Ethanol': [eth],
                    'Pressure[hPa]': [hpa],
                    'PM1.0': [pm],
                })

                missing_columns = set(model_columns) - set(input_data.columns)
                for col in missing_columns:
                    input_data[col] = 0

                input_data = input_data[model_columns]

                if st.button("Получить прогноз срабатывания датчика"):
                    try:
                        prediction = model.predict(input_data)[0]
                        formatted_prediction = "{:,.0f}".format(
                            prediction).replace(",", " ")
                        st.success(
                            f"Датчик сработал: **{'да' if formatted_prediction != '0' else 'нет'}**")
                    except Exception as e:
                        st.error(f"Ошибка при предсказании: {e}")
