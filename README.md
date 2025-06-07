# РГР: Прогнозирование стоимости недвижимости с помощью ML

## О проекте
Этот репозиторий содержит все исходные файлы для расчётно-графической работы (РГР) по дисциплине «Машинное обучение и большие данные» (МО, ОмГТУ, 2025). Тема работы:  
**«Разработка Web-приложения (дашборда) для инференса моделей ML и анализа данных»**.

В рамках проекта выполнены следующие ключевые этапы:
1. **Сбор и предобработка данных (EDA).**  
   - Ознакомление с набором данных о ценах на недвижимость (mumbai_houses_task_EDA).  
   - Очистка, заполнение пропусков, преобразование категориальных признаков.  
   - Построение базовых визуализаций (гистограммы, ящиковые диаграммы, тепловая карта корреляций, pairplot).

2. **Реализованные модели**  
   - **Полиномиальная регрессия** (Scikit-learn).  
   - **Gradient Boosting Regressor** (Scikit-learn).  
   - **XGBoost Regressor** ( XGBoost ).  
   - **Bagging Regressor** (Scikit-learn).  
   - **Stacking Regressor** (Scikit-learn).  
   - **MLPRegressor (нейронная сеть)** (Scikit-learn).

   Для каждой модели вычислены базовые метрики качества:
   - Коэффициент детерминации (R²).  
   - Средняя абсолютная ошибка (MAE).  
   - Корень из среднеквадратичной ошибки (RMSE).  


3. **Сериализация моделей.**  
   - Модели Scikit-learn сохранены с помощью `pickle` как `.pkl`.  
   - Модель XGBoost сохранена как `.json` (внутренний формат XGB).  
   - Все файлы моделей лежат в папке `models/` и доступны для загрузки при инференсе.

4. **Веб-интерфейс (Streamlit).**  
   Веб-приложение состоит из четырёх страниц (многостраничное приложение Streamlit). Оно даёт возможность:
   - **Страница 1 (General):** Информация о разработчике (ФИО, группа, тема РГР).  
   - **Страница 2 (Dataset):** Описание набора данных (предметная область, список признаков, этапы EDA).  
   - **Страница 3 (DashBoard):** Визуализации зависимостей в данных (минимум 4 разных графика с Matplotlib/Seaborn).  
   - **Страница 4 (Prediction):** Интерфейс для инференса:
     - Загрузка CSV-файла с новыми объектами недвижимости ➔ массовое предсказание.  
     - Ручной ввод признаков объекта (площадь, количество спален, ванных комнат, статус, тип, меблировка и др.) ➔ единичное предсказание.  
     - Вывод результата в понятном формате (например, цена в рупиях с разделителем тысяч).  
     - Отображение первых 10 строк загруженных данных с колонкой `predicted_price`. 
   - на **Streamlit Cloud** с развернутым веб-приложением: https://levzmeqxd4fasufn4offf8.streamlit.app/

---

## Структура репозитория
```text
ML/  
├─ README.md                       ← Документация проекта (текущий файл)  
├─ requirements.txt                ← Список Python-зависимостей для pip install   
├─ mumbai_houses_task_EDA.csv      ← Оригинальный CSV-набор
├─ nnlKausKnVY.jpg                 ← Фото студента
│    
│  
├─ models/                         ← Сохранённые модели (pickle, XGBoost .json)  
│   ├─ poly_model.pkl               ← Полиномиальная регрессия  
│   ├─ boosting_model.pkl           ← Gradient Boosting (Sklearn)  
│   ├─ xgb_model.json               ← XGBoost Regressor  
│   ├─ bagging_model.pkl            ← Bagging Regressor  
│   ├─ stacking_model.pkl           ← Stacking Regressor  
│   └─ mlp_model.pkl                ← MLPRegressor   
│  
├─ figures/                        ← Графики и таблицы метрик  
│   ├─ corr_matrix.png              ← Тепловая карта корреляций  
│   ├─ scatter_area_target.png      ← Рассеяние price vs area  
│   ├─ boxplot_Bedrooms_price.png   ← Boxplot: bedrooms vs price  
│   ├─ pairplot.png                 ← Pairplot ключевых признаков       
│   ├─ polynomial_metrics_table.png    ← Таблица метрик (Poly)          
│   ├─ boosting_metrics_table.png      ← Таблица метрик (GBR)          
│   ├─ xgb_metrics_table.png           ← Таблица метрик (XGB)          
│   ├─ bagging_metrics_table.png       ← Таблица метрик (Bagging)          
│   ├─ stacking_metrics_table.png      ← Таблица метрик (Stacking)                
│   └─ mlp_metrics_table.png           ← Таблица метрик (MLP)  
│  
├─ pages/                          ← Страницы многостраничного Streamlit-приложения  
│   ├─ 01_AboutDeveloper.py         ← Страница с информацией о разработчике  
│   ├─ 02_DatasetInfo.py            ← Страница с описанием набора данных и EDA  
│   ├─ 03_Visualizations.py         ← Страница с визуализациями (Matplotlib, Seaborn)  
│   └─ 04_Prediction.py             ← Страница с интерфейсом для предсказания  
│  
└─app.py                          ← Точка входа для Streamlit: объединяет страницы через sidebar  
                        
    
