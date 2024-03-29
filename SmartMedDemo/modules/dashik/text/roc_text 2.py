roc_table = """
Таблица значений метрик в каждой точке для данной группирующей переменной.

TP (True Positives) — верно классифицированные положительные примеры.

TN (True Negatives) — верно классифицированные отрицательные примеры.

FN (False Negatives) — положительные примеры, классифицированные как отрицательные.

FP (False Positives) — отрицательные примеры, классифицированные как положительные.

TP/{TP+FN} — чувствительность.

TN/{TN+FP}$ — специфичность.
"""

roc_roc = """
		ROC-кривая показывает зависимость количества верно классифицированных положительных примеров от количества неверно классифицированных отрицательных примеров при различных значениях порога отсечения. Чем ближе кривая к верхнему левому углу, тем выше предсказательная способность модели.
		"""

roc_table_metrics = """
        Таблица со значениями AUC и остальными метриками, выбранными ранее.

        AUC (Area Under Curve) – численный показатель площади под кривой.

        Оптимальный порог отсечения – варьируя порог отсечения, получаем то или иное разбиение на два класса, для определения оптимального порога в данном модуле используется критерий баланса между чувствительностью и специфичностью.

        Полнота – показатель, который отражает, какой процент объектов положительного класса правильно классифицировали.

        Точность – показатель, который отражает, какой процент положительных объектов (т.е. тех, что мы считаем положительными) правильно классифицирован.

        Доля верных ответов – доля (процент) объектов, на которых алгоритм выдал правильные ответы.

        F1-мера – среднее гармоническое точности и полноты.
        """

roc_inter_graph = """
        График пересечения чувствительности и специфичности для каждой зависимой переменной. Данный график наглядно показывает реализованный критерий выбора оптимального порога отсечения – критерий баланса между чувствительностью и специфичностью.
		"""

roc_comp_roc = """
        Кривая, расположенная выше и левее, свидетельствует о большей предсказательной способности модели.
        """

roc_comp_metrics = """
        Показатель AUC предназначен для сравнительного анализа нескольких моделей. Чем больше значение AUC, тем лучше модель классификации.
		"""
