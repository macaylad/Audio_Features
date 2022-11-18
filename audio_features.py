import numpy as np
import warnings
warnings.filterwarnings('ignore')
import librosa
import parselmouth
from parselmouth import praat
from nltk.sentiment import SentimentIntensityAnalyzer


def get_sentiment(text):

    sia = SentimentIntensityAnalyzer()
    if sia.polarity_scores(text)["pos"] == 0:
        sent_score = (sia.polarity_scores(text)["neg"])
    else:
        sent_score = (sia.polarity_scores(text)["neg"] / sia.polarity_scores(text)["pos"]) * .1
    return sent_score


def get_fillers(words):
    fillerwords = ['um', 'uh', 'er', 'ah', 'like', 'okay', 'right', 'you know', 'actually', 'yeah',
                   'I mean']  ##should okay be in there?
    filler = [word for word in words if word in fillerwords]

    filler_score = len(filler) / len(words) * 10

    return filler_score


def get_pitch_variability(file_name):
    sound = parselmouth.Sound(file_name)
    pitch = sound.to_pitch()

    f0min = 75
    f0max = 300

    pitch_values = pitch.selected_array['frequency']
    pitch_values[pitch_values == 0] = np.nan
    pitch_values[pitch_values > f0max] = np.nan
    pitch_values[pitch_values < f0min] = np.nan

    pitch_score = 1000 / np.nanvar(pitch_values)

    return pitch_score


def get_2formant_variabilty (file_name):
    sound = parselmouth.Sound(file_name)

    f0min = 75
    f0max = 300
    pointProcess = praat.call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

    formants = praat.call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)

    numPoints = praat.call(pointProcess, "Get number of points")
    f1_list = []
    f2_list = []
    f3_list = []
    for point in range(0, numPoints):
        point += 1
        t = praat.call(pointProcess, "Get time from index", point)
        f1 = praat.call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = praat.call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = praat.call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)

    formant2_score = 1 / np.nanvar(f2_list) * 100000

    return formant2_score


def get_av_pause_len(audio_data):
    pauses = librosa.effects.split(audio_data, top_db=20)
    num_pauses = len(pauses)

    pause_len = [pauses[i + 1][0] - pauses[i][1] for i in range(num_pauses - 1)]
    pause_score = np.nanmean(pause_len[1:-1]) / 10000

    if pause_score == np.nan:
        pause_score = []

    return pause_score


def get_percent_time_paused(audio_data, samplerate):
    total_recording_time = len(audio_data) / samplerate

    pauses = librosa.effects.split(audio_data, top_db=20)
    num_pauses = len(pauses)

    pause_len = [(pauses[i + 1][0] - pauses[i][1]) / samplerate for i in range(num_pauses - 1)]
    mean_pause_len = np.mean(pause_len)
    pause_var = np.var(pause_len)

    total_pause_len = sum(pause_len)
    percent_paused_score = total_pause_len / total_recording_time

    return percent_paused_score


def get_speech_rate(words,audio_data,samplerate):
    rate_of_speech = len(words) / (len(audio_data) / samplerate)
    return rate_of_speech


def get_mfcc2(audio_data, samplerate):

    mfccs = librosa.feature.mfcc(y=audio_data, sr=samplerate, n_mfcc=10)
    mfcc_2 = mfccs[1]
    mfcc_2_score = np.nanmean(mfcc_2) / 100

    return mfcc_2_score


def get_intensity(file_name):

    sound = parselmouth.Sound(file_name)
    intensity = sound.to_intensity()

    intensity_values = intensity.values.T
    intensity_values[intensity_values == 0] = np.nan
    intensity_score=np.nanmean(intensity)/10

    return intensity_score
