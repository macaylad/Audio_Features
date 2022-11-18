import process_audio
import audio_features


def get_loneliness_index(file_name_m4a, verbose=bool):
    file_name, audio_data, samplerate = process_audio.convert_extract_audio_data(file_name_m4a)
    text, words = process_audio.get_text_from_speech(file_name, audio_data)

    sent_score = audio_features.get_sentiment(text)
    filler_score = audio_features.get_fillers(words)
    pitch_score = audio_features.get_pitch_variability(file_name)
    formant2_score = audio_features.get_2formant_variabilty(file_name)
    pause_score = audio_features.get_av_pause_len(audio_data)

    loneliness_index = sent_score + filler_score + pitch_score + formant2_score + pause_score

    if verbose:
        return loneliness_index, sent_score, filler_score, pitch_score, formant2_score, pause_score, text
    else:
        return loneliness_index


def get_depression_index(file_name_m4a, verbose=bool):
    file_name, audio_data, samplerate = process_audio.convert_extract_audio_data(file_name_m4a)
    text, words = process_audio.get_text_from_speech(file_name, audio_data)

    sent_score = audio_features.get_sentiment(text)
    filler_score = audio_features.get_fillers(words)
    pitch_score = audio_features.get_pitch_variability(file_name)
    pause_score = audio_features.get_av_pause_len(audio_data)
    mfcc2_score = audio_features.get_mfcc2(audio_data, samplerate)
    rate_of_speech = audio_features.get_speech_rate(words, audio_data, samplerate)

    depression_index = sent_score + filler_score + pitch_score + mfcc2_score + pause_score + rate_of_speech

    if verbose:
        return depression_index, sent_score, filler_score, pitch_score, mfcc2_score, pause_score, text
    else:
        return depression_index


def get_anxiety_index(file_name_m4a, verbose=bool):
    file_name, audio_data, samplerate = process_audio.convert_extract_audio_data(file_name_m4a)
    text, words = process_audio.get_text_from_speech(file_name, audio_data)

    pitch_score = audio_features.get_pitch_variability(file_name) ## same as f0?


print(get_depression_index('/Users/macayla.donegan/Documents/messagingFolder_NA1000019627/11-3-2022.m4a',verbose=True))
print(get_loneliness_index('/Users/macayla.donegan/Documents/messagingFolder_NA1000019627/11-3-2022.m4a',verbose=True))

