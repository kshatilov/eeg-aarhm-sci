import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import mne
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from scipy.signal import butter, sosfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mne.io import read_raw_gdf
import numpy as np
from EEGModels import EEGNet
import warnings
from tensorflow_core.python.keras.callbacks import ModelCheckpoint
from tensorflow_core.python.keras.utils.np_utils import to_categorical

warnings.filterwarnings("ignore", category=DeprecationWarning)


class Constants:
    t_max = 1.5
    t_min = -.1
    subj = [
        'P01',
        'P06',
        'P07',
    ]
    events = {
        '776': "supination class cue",
        '777': "pronation class cue",
        '779': "hand open class cue",
        '925': "palmar grasp class cue",
        '926': "lateral grasp class cue",
    }
    runs = [
        3, 4, 5, 6, 7, 10, 11, 12, 13
    ]
    channels = 64


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y


class KEEGclf:

    def __init__(self, subj) -> None:
        super().__init__()
        self.files = []
        self.X = None
        self.y = None
        self.subj = subj

    def read_raw(self):
        subj = self.subj
        for i in Constants.runs:
            self.files.append(read_raw_gdf(
                f'data\\{subj}\\{subj} Run {i}.gdf', stim_channel=None, verbose=False))

    def extract_epochs(self):
        epochs_list = []
        for file in self.files:
            event, labels = mne.events_from_annotations(file, verbose=False)
            selected_labels = {x: labels[x] for x in Constants.events}

            epochs = mne.Epochs(file, event,
                                selected_labels,
                                tmin=Constants.t_min,
                                tmax=Constants.t_max,
                                baseline=None,
                                verbose=False,
                                event_repeated='merge',
                                preload=True)
            epochs_list.append(epochs)

        result_epochs = mne.concatenate_epochs(epochs_list)
        self.X = result_epochs.get_data()
        self.y = result_epochs.events[:, -1]

    def encode_labels(self):
        encoder = LabelEncoder()
        encoder.fit(self.y)
        self.y = encoder.transform(self.y)

    def reshape_for_cnn(self, x):
        return x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])

    def narrow_channels(self):
        a = 24
        b = 32
        self.X = np.asarray([trial[a:b, :] for trial in self.X])

    def bandpass(self, _fs=256, _lowcut=0.3, _highcut=3):

        self.X = [[butter_bandpass_filter(row, _lowcut, _highcut, _fs) for row in trial] for trial in self.X]
        self.X = np.asarray(self.X)

    def stats(self):
        for label in range(len(Constants.events)):
            print(f"{label}: {len(self.y[self.y == label])}")

    def run_cnn(self):
        self.read_raw()
        self.extract_epochs()
        self.encode_labels()
        self.bandpass(_lowcut=0.1, _highcut=100)
        self.narrow_channels()
        self.filter(_min=-100, _max=300)

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        X_train = self.reshape_for_cnn(X_train)
        X_test = self.reshape_for_cnn(X_test)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        model = EEGNet(nb_classes=len(Constants.events), Chans=self.X.shape[1], Samples=self.X.shape[2])

        model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        checkpointer = ModelCheckpoint(filepath='./checkpoint.h5', verbose=False,
                                       save_best_only=True)
        model.fit(X_train, y_train, batch_size=6, epochs=20,
                  verbose=False, validation_data=(X_test, y_test), callbacks=[checkpointer])
        model.load_weights('./checkpoint.h5')
        y_pred = model.predict(X_test)
        y_pred = y_pred.argmax(axis=-1)
        acc = np.mean(y_pred == y_test.argmax(axis=-1))
        print(f"{self.subj}: {acc}")

    def flatten(self):
        X = [np.asarray(trial).flatten(order='F') for trial in self.X]
        self.X = np.asarray(X)

    def filter(self, _min=-400, _max=400):
        X = []
        y = []
        for i, trial in enumerate(self.X):

            bad_trial = False
            for row in trial:
                for value in row:
                    if value > _max or value < _min:
                        bad_trial = True
                        break

            if not bad_trial:
                X.append(trial)
                y.append(self.y[i])

        self.X = np.asarray(X)
        self.y = np.asarray(y)

    def run_lda(self):
        self.read_raw()
        self.extract_epochs()
        self.encode_labels()
        self.narrow_channels()
        self.bandpass(_lowcut=0.1, _highcut=100)
        self.filter(_min=-100, _max=300)
        self.bandpass()
        self.flatten()
        self.stats()

        clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=0.1)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=42,
                                                            shuffle=True)
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        acc = accuracy_score(y_true=y_test, y_pred=predicted)
        print(f"{self.subj}: {acc}")


if __name__ == '__main__':
    print("LDA")
    for subj in Constants.subj:
        clf = KEEGclf(subj=subj)
        clf.run_lda()

    print("CNN")
    for subj in Constants.subj:
        clf = KEEGclf(subj=subj)
        clf.run_cnn()
