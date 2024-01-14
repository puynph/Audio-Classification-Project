import os
import librosa
import math
import json


DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')
print(DATASET_PATH)
JSON_PATH = "data.json"
SAMPLE_RATE = 22050
DURATION = 30  # secs
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION


def mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, segments=5):
    # store data into a dictionary
    data = {
        "mapping": [],  # map genre to integer value
        "mfcc": [],  # training feature
        "label": []  # output
    }

    # for segment processing
    samples_per_segment = int(SAMPLES_PER_TRACK / segments)
    uniform_mfcc_length = math.ceil(samples_per_segment / hop_length)

    # loop through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_part = os.path.split(dirpath)  # genres/blues
            label = dirpath_part[-1]
            data["mapping"].append(label)
            print("\nProcessing: {}".format(label))

            # audio in a specific genre
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                # process short segments in a track - mfcc extracting
                for seg in range(segments):
                    start = samples_per_segment * seg
                    end = samples_per_segment + start
                    mfcc = librosa.feature.mfcc(signal[start:end],
                                                n_mfcc=n_mfcc,
                                                sr=sr,
                                                n_fft=n_fft,
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # ensure uniform shape of mfcc for each segment - cut or sero pad
                    if len(mfcc) == uniform_mfcc_length:
                        data["mfcc"].append(mfcc.tolist())
                        data["label"].append(i-1)  # ignore the first loop
                        print(f"{file_path}, segment: {seg + 1}")

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__ == "__main__":
    mfcc(DATASET_PATH, JSON_PATH, segments=10)
