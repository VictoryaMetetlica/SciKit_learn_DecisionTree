from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Скачайте набор данных с тремя переменными: sex, exang, num. Представьте, что при помощи дерева решений мы хотим
# классифицировать есть или нет у пациента заболевание сердца (переменная num), основываясь на двух признаках:
# пол (sex) и наличие/отсутсвие стенокардии (exang). Обучите дерево решений на этих данных, используйте entropy
# в качестве критерия.

    # файл для работы
dt = pd.read_csv('train_data_tree.csv')
print('\nДанные\n', dt.head(3))
y = dt.num
X = dt[['sex','exang']]
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf.fit(X, y)
tree.plot_tree(clf, filled=True)
plt.show()

# вам даны 2 numpy эррея с измеренными признаками ирисов и их принадлежностью к виду. Сначала попробуем
# способ с разбиением данных на 2 датасэта. Используйте функцию train_test_split для разделения имеющихся
# данных на тренировочный и тестовый наборы данных, 75% и 25% соответственно. # Затем создайте дерево dt с
# параметрами по умолчанию и обучите его на тренировочных данных, а после предскажите классы, к которым
# принадлежат данные из тестовой выборки, сохраните результат предсказаний в переменную predicted.


iris = load_iris()
X = iris.data
y = iris.target
	# Сплит данных на тренировочную и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
	# Инициализация и обучение Дерева
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
	# Предсказание результата на тестовой выборке
predicted = dt.predict(X_test)
print('\n predicted \n', predicted)

# осуществите перебор всех деревьев на данных ириса по следующим параметрам:
#     • максимальная глубина - от 1 до 10 уровней
#     • минимальное число проб для разделения - от 2 до 10
#     • минимальное число проб в листе - от 1 до 10
# и сохраните в переменную best_tree лучшее дерево. Переменную с GridSearchCV назовите search

iris = load_iris()
X = iris.data
y = iris.target
dt = tree.DecisionTreeClassifier()
parametrs = {'criterion': ['entropy'],
             'max_depth': range(1, 10),
             'min_samples_split': range(2, 10),
             'min_samples_leaf': range(1, 10)}
search = GridSearchCV(dt, parametrs)
search.fit(X, y)
best_tree = search.best_estimator_
print('\n best_tree1 \n', best_tree)

# Осуществим поиск по тем же параметрам что и в предыдущем задании с помощью RandomizedSearchCV
#     • максимальная глубина - от 1 до 10 уровней
#     • минимальное число проб для разделения - от 2 до 10
#     • минимальное число проб в листе - от 1 до 10
# Cохраните в переменную best_tree лучшее дерево. Переменную с RandomizedSearchCV назовите search

iris = load_iris()
X = iris.data
y = iris.target
dt = tree.DecisionTreeClassifier()
parametrs = {'criterion': ['entropy'],
             'max_depth': range(1, 10),
             'min_samples_split': range(2, 10),
             'min_samples_leaf': range(1, 10)}
search = RandomizedSearchCV(dt, parametrs)
search.fit(X, y)
best_tree = search.best_estimator_
print('\n best_tree 2\n', best_tree)