from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

    # файл для работы
data = pd.read_csv('dogs.csv', index_col=0, encoding='UTF-8')
data.head()
    # Создаем дерево для обучения
print('Обозначаем дерево для дальнейшей работы')
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
print('Передаем данные Х - фичи для обработки, у - данные проверки')
X = data.iloc[:, :3]
y = data.iloc[:, 3]
print('Создаем классификатор дерева решений из обучающего набора (X, y)')
clf.fit(X, y)
print('Выводим дерево')
tree.plot_tree(clf, feature_names=data.columns)
plt.show()


# В нашем Big Data датасэте появились новые наблюдения! Давайте немного посчитаем энтропию, чтобы лучше понять,
# формализуемость разделения на группы.

    # файл для работы
data = pd.read_csv('cats.csv', index_col=0, encoding='UTF-8')
data.head()
    # Создаем дерево для обучения
print('Обозначаем дерево для дальнейшей работы')
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
print('Передаем данные Х - фичи для обработки, у - данные проверки')
X = data[["Шерстист"]]
y = data["Вид"]
print('Создаем классификатор дерева решений из обучающего набора (X, y)')
clf.fit(X, y)
print('Выводим дерево')
print('Энтропия при разделении по фиче Шерстист в группах, где Шерстист равно 0 и 1 соответственно, составляет ')
tree.plot_tree(clf, feature_names=data.columns)
plt.show()
print('Энтропия при разделении по фиче Гавкает в группах, где Гавкает равно 0 и 1 соответственно, составляет ')
X = data[["Гавкает"]]
y = data["Вид"]
print('Создаем классификатор дерева решений из обучающего набора (X, y)')
clf.fit(X, y)
print('Выводим дерево')
tree.plot_tree(clf, feature_names=data.columns)
plt.show()

# Ещё немного арифметики - посчитаем Information Gain по данным из предыдущего задания. Впишите через пробел
# округлённые до 2-ого знака значения IG для фичей Шерстист, Гавкает и Лазает по деревьям. Десятичным
# разделителем в данном задании является точка.

print('рассчитаем IG')
e = 0.97
print(1/10 * 0 + 9/10 * 0.97 * 0.99)
print(5/10 * 0 + 5/10 * 0.97 * 0.72)
print(4/10 * 0 + 6/10 * 0)

# Мы собрали побольше данных о котиках и собачках, и готовы обучить нашего робота их классифицировать!
# Скачайте json  тренировочный датасэт и обучите на нём Decision Tree. После этого скачайте датасэт из задания
# и предскажите какие наблюдения к кому относятся. Введите число собачек, допускается погрешность.

df_tr = pd.read_csv(r'dogs_n_cats.csv', index_col=0, encoding='UTF-8')
print('выводим наименование столбцов:')
print(df_tr.columns)
    # Удаляем целевой столбец:
X = df_tr.drop(['Вид'], axis=1)
y = df_tr['Вид']
    # Обучение.
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X, y)
    # Определение количество собачек в тестовой выборке - скачиваем
df_ts = pd.read_json(r'/home/nikolay/Documents/Projects_V_Metelica/SciKit learn/Task/dataset_209691_15.txt', encoding='UTF-8')
    #  - проверяем соответствие столбцов тестовой и тренировочной
df_tr[:2]
df_tr[:2]
    # - если порядок не совпадает, то приводим к единому текущий порядок столбцов[b, c, d, a]
X_ts = df_ts[['Высота', 'Шерстист', 'Гавкает', 'Лазает по деревьям']]
result = clf.predict(X_ts)
print(pd.Series(result)[result == 'собачка'].count())