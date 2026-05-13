import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import joblib
from datetime import datetime

label_encoder = LabelEncoder()
scaler = StandardScaler()

def isna_check(df_col):
    return df_col.isna().sum()

def prepre_data(df):
    df.drop(columns = ['Name', 'PassengerId', 'Ticket', 'Cabin', 'Fare'], inplace=True)
    df['Sex'] = label_encoder.fit_transform(df['Sex'])
    
    if isna_check(df['Age']) > 0:
        df['Age'] = df['Age'].fillna(df['Age'].mean())
        
    if isna_check(df['Embarked']) > 0:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        
    df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
    x = df.drop(columns = ['Survived'])
    y = df['Survived']
    return x, y

class TestSetEvaluationCallback(Callback):
    def __init__(self, x_test, y_test):
        super(TestSetEvaluationCallback, self).__init__()
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs=None):
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"\nEpoch {epoch+1}: Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


def main():
    df = pd.read_csv('./dataset/train.csv')
    validation_df = pd.read_csv('./dataset/validation.csv')
    test_df = pd.read_csv('./dataset/test-titanic.csv')

    X_train, y_train = prepre_data(df)
    X_val, y_val = prepre_data(validation_df)
    X_test, y_test = prepre_data(test_df)

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = './model'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'model_{timestamp}.h5')

    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)

    label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
    joblib.dump(label_encoder, label_encoder_path)

    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint_callback = ModelCheckpoint(
        filepath=model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
    )
    
    test_evaluation_callback = TestSetEvaluationCallback(X_test_scaled, y_test)

    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[checkpoint_callback, early_stopping_callback, test_evaluation_callback]
    )

    final_test_loss, final_test_accuracy = model.evaluate(X_test_scaled, y_test)
    print(f"Final Test Loss: {final_test_loss}, Final Test Accuracy: {final_test_accuracy}")

if __name__ == "__main__":
    main()
