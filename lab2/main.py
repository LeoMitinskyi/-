import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Считываем набор данных
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

train_data.head()
test_data.head()

# Проверяем нет ли в тестовых данных пустых значений значений (Nan)
def missing_value_checker(data):
    lst = []
    for feature, content in data.items():
        if data[feature].isnull().values.any():
            s = data[feature].isna().sum()

            t = data[feature].dtype

            print(f'{feature}: {s}, type: {t}')

            lst.append(feature)
    print(lst)

    print(len(lst))


missing_value_checker(test_data)

test_edited = test_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
train_edited = train_data.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)


def nan_filler(data):
    for label, content in data.items():
        if pd.api.types.is_numeric_dtype(content):
            data[label] = content.fillna(content.mean())
        else:
            data[label] = content.astype("category").cat.as_ordered()
            data[label] = pd.Categorical(content).codes + 1


nan_filler(test_edited)
nan_filler(train_edited)

missing_value_checker(test_edited)
missing_value_checker(train_edited)
print(train_edited.shape, test_edited.shape)
test_edited.info()
train_edited.info()

X = train_edited.drop('SalePrice', axis=1)
y = train_edited['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
print(X_train.shape, test_edited.shape)

# Построение и обучение модели
model = tf.keras.Sequential([
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1)
])
tf.random.set_seed(40)

def compile_model(epochs, loss, metrics, batch_size=50):
    model.compile(optimizer='adam',
                loss=tf.keras.losses.MeanAbsoluteError(),
                metrics=metrics)
    history1 = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)
    
    pd.DataFrame(history1.history).plot()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    print(history1.history)
    scores = model.evaluate(X_val, y_val, verbose=1)
    print('scores', scores)
    model.compile(optimizer='adam',
                loss=loss)
    history2 = model.fit(X_train, y_train, batch_size=10, epochs=epochs)
    pd.DataFrame(history2.history).plot()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

# Оценка полученных результатов для разного количесвтва эпох
for i in range(10, 60, 10):
    compile_model(i, tf.keras.losses.MeanSquaredLogarithmicError(), ['mae']) # square(log(true -1) -log(pred -1
)
    


preds = model.predict(test_edited)
print('preds', preds)
preds

output = pd.DataFrame(
{
    'Id': test_data['Id'],
    'SalePrice': np.squeeze(preds)
})
output