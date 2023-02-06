import numpy as np
import warnings

warnings.filterwarnings('ignore')

import librosa
import parselmouth
from parselmouth import praat
from nltk.sentiment import SentimentIntensityAnalyzer

import speech_recognition as sr
import librosa
from pydub import AudioSegment
import string


class ProcessAudio:
    def __init__(self, file_name_m4a):
        self.filename_m4a = file_name_m4a

    def convert_extract_audio_data(self):
        """convert m4a files to wav files for audio processing in python"""

        fn = self.file_name_m4a[:-4]

        sound = AudioSegment.from_file(self.file_name_m4a, format='m4a')
        sound = sound.export(fn + '.wav', format='wav')

        file_name = fn + '.wav'
        audio_data, samplerate = librosa.load(file_name)
        return file_name, audio_data, samplerate

    def get_text_from_speech(self, file_name, audio_data):
        """ Uses googles speech recognition algorithm to convert speech to text. file must be a .wav file """
        r = sr.Recognizer()
        with sr.AudioFile(file_name) as source:
            # listen for the data (load audio to memory)
            r.adjust_for_ambient_noise(source)
            data = r.record(source)
            # recognize (convert from speech to text)
            try:
                text = r.recognize_google(data)
            except:
                print('Could not convert speech to text')

            if len(text) > 0 and max(audio_data) > 0.1:  ## very quiet recordings wont work here
                words = text.split()
                # remove punctuation from each word
                table = str.maketrans('', '', string.punctuation)
                text_list = [w.translate(table) for w in words]
                text = ' '.join(text_list)
            return text, words


class AudioFeatures:
    def __init__(self, file_name, audio_data, samplerate, text, words):
        self.filename = file_name
        self.audio_data = audio_data
        self.samplerate = samplerate
        self.text = text
        self.words = words

    def get_sentiment(self):
        sia = SentimentIntensityAnalyzer()
        if sia.polarity_scores(self.text)["pos"] == 0:
            sent_score = (sia.polarity_scores(self.text)["neg"])
        else:
            sent_score = (sia.polarity_scores(self.text)["neg"] / sia.polarity_scores(self.text)["pos"]) * .1
        return sent_score

    def get_fillers(self):
        fillerwords = ['um', 'uh', 'er', 'ah', 'like', 'okay', 'right', 'you know', 'actually', 'yeah',
                       'I mean']  ##should okay be in there?
        filler = [word for word in self.words if word in fillerwords]

        filler_score = len(filler) / len(self.words) * 10

        return filler_score

    def get_pitch_variability(self):
        sound = parselmouth.Sound(self.file_name)
        pitch = sound.to_pitch()

        f0min = 75
        f0max = 300

        pitch_values = pitch.selected_array['frequency']
        pitch_values[pitch_values == 0] = np.nan
        pitch_values[pitch_values > f0max] = np.nan
        pitch_values[pitch_values < f0min] = np.nan

        pitch_score = 1000 / np.nanvar(pitch_values)

        return pitch_score

    def get_2formant_variabilty(self):
        sound = parselmouth.Sound(self.file_name)

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


    def get_av_pause_len(self):
        pauses = librosa.effects.split(self.audio_data, top_db=20)
        num_pauses = len(pauses)

        pause_len = [pauses[i + 1][0] - pauses[i][1] for i in range(num_pauses - 1)]
        pause_score = np.nanmean(pause_len[1:-1]) / 10000

        if pause_score == np.nan:
            pause_score = []

        return pause_score


    def get_percent_time_paused(self):
        total_recording_time = len(self.audio_data) / self.samplerate

        pauses = librosa.effects.split(self.audio_data, top_db=20)
        num_pauses = len(pauses)

        pause_len = [(pauses[i + 1][0] - pauses[i][1]) / samplerate for i in range(num_pauses - 1)]
        mean_pause_len = np.mean(pause_len)
        pause_var = np.var(pause_len)

        total_pause_len = sum(pause_len)
        percent_paused_score = total_pause_len / total_recording_time

        return percent_paused_score

    def get_speech_rate(self):
        rate_of_speech = len(self.words) / (len(self.audio_data) / self.samplerate)
        return rate_of_speech


    def get_mfcc2(self):
        mfccs = librosa.feature.mfcc(y=self.audio_data, sr=self.samplerate, n_mfcc=10)
        mfcc_2 = mfccs[1]
        mfcc_2_score = np.nanmean(mfcc_2) / 100

        return mfcc_2_score

    def get_intensity(self):
        sound = parselmouth.Sound(self.file_name)
        intensity = sound.to_intensity()

        intensity_values = intensity.values.T
        intensity_values[intensity_values == 0] = np.nan
        intensity_score = np.nanmean(intensity) / 10

        return intensity_score

    def get_f0(self):
        fft_spectrum = np.fft.rfft(self.audio_data)
        freq = np.fft.rfftfreq(self.audio_data.size, d=1. / self.samplerate)
        fft_spectrum_abs = np.abs(fft_spectrum)

    ##get rid of 60 hz noise
        for i, f in enumerate(freq):
            if f < 62 and f > 58:  # (1)
                fft_spectrum[i] = 0.0
            if f < 21 or f > 20000:  # (2)
                fft_spectrum[i] = 0.0

        f0_max = max(fft_spectrum_abs[np.where(freq < 3000)])

        if freq[np.where(fft_spectrum_abs == f0_max)][0] > 0:
            f0_score = 100 / freq[np.where(fft_spectrum_abs == f0_max)][0]
        else:
            f0_score = np.nan
        return f0_score


    def get_shimmer(self):
        sound = parselmouth.Sound(self.file_name)
        pitch = sound.to_pitch()
        pointProcess = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)...", 75, 600)
        pulse = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")

        parselmouth.praat.call([sound, pitch, pulse], "Voice report", 0, 0, 75, 600, 1.3, 1.6, 0.03, 0.45)

        shimmer_score = .1 / parselmouth.praat.call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3,
                                                1.6)
        return shimmer_score


    def get_number_of_pauses(self):
        total_recording_time = len(self.audio_data) / self.samplerate

        pauses = librosa.effects.split(self.audio_data, top_db=20)
        pause_score = len(pauses) / (total_recording_time)
        return pause_score

    def get_hnr(self):
        sound = parselmouth.Sound(self.file_name)
        harmonicity = sound.to_harmonicity()
        harmonicity.values[harmonicity.values == -200] = np.nan
        hnr_score = np.nanmean(harmonicity) * .1

        return hnr_score
