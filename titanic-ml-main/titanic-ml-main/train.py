import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras
import joblib
from datetime import datetime

label_encoder = LabelEncoder()

def prepare_data(data):
    data.drop(columns=['Name', 'PassengerId', 'Ticket', 'Cabin', 'Fare'], inplace=True)
    data.dropna(subset=['Age'], inplace=True)
    data['Sex'] = label_encoder.fit_transform(data['Sex'])
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0]) 
    data['Embarked'] = label_encoder.fit_transform(data['Embarked'])
    x = data.drop(columns=['Survived'])
    y = data['Survived']
    return x, y

train_data = pd.read_csv("dataset/train.csv")
validation_data = pd.read_csv("dataset/validation.csv")
test_data = pd.read_csv("dataset/test.csv")

x_train, y_train = prepare_data(train_data)
x_val, y_val = prepare_data(validation_data)
x_test, y_test = prepare_data(test_data) 

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_val_scaled = scaler.transform(x_val)
x_test_scaled = scaler.transform(x_test)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f'models/titanic_model_{timestamp}'
os.makedirs(model_dir, exist_ok=True)

scaler_path = os.path.join(model_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)

label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
joblib.dump(label_encoder, label_encoder_path)

model = keras.Sequential()
model.add(keras.layers.Input(shape=(x_train_scaled.shape[1],)))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    os.path.join(model_dir, 'best_model.keras'), 
    monitor='val_accuracy',         
    save_best_only=True,           
    mode='max',                                          
)

class TestSetEvaluationCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test):
        super(TestSetEvaluationCallback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"\nEpoch {epoch+1}: Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

test_evaluation_callback = TestSetEvaluationCallback(x_test_scaled, y_test)

history = model.fit(
    x_train_scaled, y_train,
    validation_data=(x_val_scaled, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[checkpoint_callback, test_evaluation_callback]
)

final_test_loss, final_test_accuracy = model.evaluate(x_test_scaled, y_test)
print(f"Final Test Loss: {final_test_loss}, Final Test Accuracy: {final_test_accuracy}")
