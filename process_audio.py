import speech_recognition as sr
import librosa
from pydub import AudioSegment
import string


def convert_extract_audio_data(file_name_m4a):
    fn = file_name_m4a[:-4]

    sound = AudioSegment.from_file(file_name_m4a, format='m4a')
    sound = sound.export(fn + '.wav', format='wav')

    file_name = fn + '.wav'
    audio_data, samplerate = librosa.load(file_name)
    return file_name, audio_data, samplerate


def get_text_from_speech(file_name, audio_data):
    ### Uses googles speech recognition algorithm to convert speech to text. file must be a .wav file
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
