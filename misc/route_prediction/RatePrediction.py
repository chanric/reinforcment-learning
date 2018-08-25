import numpy as np



_TLS_SLOWDOWN = 1.3
R1_AVE_RSP_TIME = 2.1
R2_AVE_RSP_TIME = 1
R3_AVE_RSP_TIME = 1.5
R4_AVE_RSP_TIME = 2.5

ROUTE_TO_ACTUAL_RATES = [R1_AVE_RSP_TIME, R2_AVE_RSP_TIME, R3_AVE_RSP_TIME, R4_AVE_RSP_TIME]


NOISE_STD = .02

STD_PERCENT = 0.0000000000



def generate_random_ts(route, TLS=False):
    route_rate = ROUTE_TO_ACTUAL_RATES[route]
    rate = route_rate + _TLS_SLOWDOWN if TLS else route_rate
    scale = STD_PERCENT*rate
    rate += np.random.normal(loc=0, scale=NOISE_STD)
    return rate #np.random.normal(loc=rate, scale=scale)

def generate_random_ts_from_X(a):
    return generate_random_ts(a[0], a[1])


def gen_X_y(samples=5):
    random_rates = np.random.choice((0,1,2,3), size=(samples, 1))
    random_TLS = np.random.choice((0, 1), size=(samples, 1))
    X = np.concatenate((random_rates, random_TLS), axis=1)
    y = np.apply_along_axis(generate_random_ts_from_X, 1, X)
    return (X, y)


# create are raw sample data
(X_full_raw, y_full_raw) = gen_X_y(samples=25000)
#keep raw data as reference
X = X_full_raw
y = y_full_raw

# Encoding categorical data (needed if we want to routes as a category
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()

X[:, 0] = labelencoder_X_1.fit_transform(X[:, 0])
onehot_encoder = OneHotEncoder(categorical_features=[0])
X = onehot_encoder.fit_transform(X).toarray()

#split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


import keras
from keras.models import Sequential
from  keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(units=40, activation = 'relu', input_dim=np.shape(X_train)[1]))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=20, activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=10, activation='relu'))
classifier.add(Dropout(0.1))
#last layer is relu because we are doing a prediction?
classifier.add(Dense(units=1, activation = 'relu'))
classifier.compile(optimizer='adam', loss ='mean_squared_error', metrics=['mse', 'mae', 'mape', 'cosine'])
classifier.fit(X_train, y_train, batch_size=200, epochs=100)


y_pred = classifier.predict(X_test)
y_test = y_test.reshape(np.shape(y_test)[0], 1)
print(np.mean(np.square(y_pred-y_test)))

