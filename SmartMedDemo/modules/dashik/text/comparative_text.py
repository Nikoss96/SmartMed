markdown_student_p_value = """
    Если p < 0.05, нулевая гипотеза отвергается, принимается альтернативная, различия обладают статистической значимостью и носят системный характер.\n
    Если p ≥ 0.05, принимается нулевая гипотеза, различия не являются статистически достоверными и носят случайный характер.
"""

markdown_student_variables_with_group = """
    Группирующая переменная - переменная, используемая для разбиения независимой переменной на группы. Представлена в виде бинарной переменной, например, пол, группа и т.д.\n
    Независимая переменная представляет набор количественных, непрерывных значений, например, возраст пациента, уровень лейкоцитов и т.д.
"""

markdown_student_variables = """
    Независимая переменная представляет набор количественных, непрерывных значений, например, возраст пациента, уровень лейкоцитов и т.д.
"""

markdown_student_ind_head = """
    Для применения t-критерия Стьюдента необходимо, чтобы исходные данные имели нормальное распределение.\n
    Данный статистический метод служит для сравнения двух несвязанных между собой групп. Примеры сравниваемых величин: возраст в основной и контрольной группе, содержание глюкозы в крови пациентов, принимавших препарат или плацебо.
"""

markdown_student_depend_head = """
    Для применения t-критерия Стьюдента необходимо, чтобы исходные данные имели нормальное распределение.\n
    Данный метод используется для сравнения зависимых групп пациентов - результатов, полученных для одних и тех же исследуемых (например, частота сердечных сокращений до и после приема препарата, содержание лейкоцитов в крови пациентов до и после лечения).
"""

markdown_mann_whitney_head = """
    Данный статистический метод служит для сравнения двух несвязанных между собой групп в случае, когда распределение в выборках не является нормальным или неизвестно. Примеры сравниваемых величин: возраст в основной и контрольной группе, содержание глюкозы в крови пациентов, принимавших препарат или плацебо.
    
"""

markdown_wilcoxon_head = """
    Данный метод используется для сравнения зависимых групп пациентов в случае, когда распределение в выборках не является нормальным или неизвестно - результатов, полученных для одних и тех же исследуемых (например, частота сердечных сокращений до и после приема препарата, содержание лейкоцитов в крови пациентов до и после лечения).
"""

markdown_odds_relations_head = """
    Отношение шансов показывает, насколько отсутствие или наличие определённого исхода связано с присутствием или отсутствием определённого фактора в каждой из групп пациентов.
"""

markdown_odds_relations = """
    Если p < 0.05, делается вывод о статистической значимости выявленной связи между фактором и исходом.\n
    Если p ≥ 0.05, делается вывод об отсутствии статистической значимости связи между фактором и исходом.\n
    Если OR > 1, то шансы обнаружить фактор риска больше в группе с наличием исхода. Фактор имеет прямую связь с вероятностью наступления исхода.\n
    OR показывает, во сколько раз шансы исхода в основной группе выше, чем в контрольной.\n
    Если OR < 1, то шансы обнаружить фактор риска больше во второй группе. Фактор имеет обратную связь с вероятностью наступления исхода.\n
    1/OR показывает, во сколько раз шансы исхода в контрольной группе выше, чем в основной.\n
    Если OR = 1, то шансы обнаружить фактор риска в сравниваемых группах одинакова. Фактор не оказывает никакого воздействия на вероятность исхода.\n
    \n
    Пример:\n
    Первая группа: 18 пациентов с  заболеванием на 100 курящих пациентов\n
    Вторая группа: 6 пациентов с заболеванием на 100 некурящих пациентов\n
    OR = (18 * 94) / (6 * 82) = 3,44\n
    Шанс встретить это заболевание среди курильщиков в 3,44 раза выше, чем среди некурящих\n
"""

markdown_risk_relations_head = """
    Относительный риск используется для сравнения риска заболеваемости в двух разных группах пациентов. Это соотношение абсолютного риска события в одной группе по сравнению с абсолютным риском события в другой группе.\n
"""

markdown_risk_relations = """
    RR используется для сравнения вероятности исхода в зависимости от наличия фактора риска.\n
    Если p < 0.05, делается вывод о статистической значимости выявленной связи между фактором и исходом.\n
    Если p ≥ 0.05, делается вывод об отсутствии статистической значимости связи между фактором и исходом.\n
    Если RR > 1, то фактор повышает частоту исходов (прямая связь).\n
    Если RR < 1, то вероятность исхода снижается при воздействии фактора (обратная связь).\n
    Если RR = 1, то исследуемый фактор не влияет на вероятность исхода.\n
    \n
    Пример:\n
    Абсолютный риск развития заболевания у некурящих составляет 6 из 100: 6:100 = 0,06 или 6%.\n
    Абсолютный риск развития этого заболевания у курильщиков составляет 18 из 100: 18:100 = 0,18 или 18%.\n
    Относительный риск заболеваемости среди курящих людей в 0,18:0,06 = 3 раза выше, чем у некурящих.\n
"""

markdown_kolm_smirn = """
    Если p < 0.05, нулевая гипотеза отвергается, принимается альтернативная, выборка не имеет нормального распределения.\n
    Если p ≥ 0.05, принимается нулевая гипотеза, выборка имеет нормальное распределение.
"""

markdown_kolm_smirn_head = """
    Критерий согласия Колмогорова-Смирнова предназначен для проверки гипотезы о принадлежности выборки некоторому закону распределения (нормальному).
"""

markdown_se_sep = """
    Чувствительность метода - вероятность того, что результат исследования будет положительным при наличии заболевания.\n
    Sensitivity = TP / (TP + FN)\n
    Специфичность - вероятность того, что результат будет отрицательным при отсутствии заболевания.\n
    Specificity = TN / (TN + FP)\n
    TP - истинный положительный результат: больные люди правильно идентифицированы как больные;\n
    TN - истинно отрицательный: здоровые люди правильно идентифицированы как здоровые;\n
    FP - ложноположительный результат: здоровые люди ошибочно идентифицированы как больные;\n
    FN - ложноотрицательный результат: больные ошибочно идентифицированы как здоровые.
"""

markdown_chi_2_head = """
    Хи квадрат позволяет проводить анализ четырехпольных таблиц, когда и фактор, и исход являются бинарными переменными, то есть имеют только два возможных значения (например, мужской или женский пол, наличие или отсутствие определенного заболевания в анамнезе.)
"""

markdown_chi_2 = """
    Если p < 0.05, нулевая гипотеза отвергается, принимается альтернативная, различия обладают статистической значимостью и носят системный характер.\n
    Если p ≥ 0.05, принимается нулевая гипотеза, различия не являются статистически достоверными и носят случайный характер.
"""
