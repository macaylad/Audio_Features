import audio_features


def get_loneliness_index(file_name_m4a, verbose=bool):
    file_name, audio_data, samplerate = audio_features.ProcessAudio.convert_extract_audio_data(file_name_m4a)
    text, words = audio_features.ProcessAudioprocess_audio.get_text_from_speech(file_name, audio_data)

    sent_score = audio_features.AudioFeatures.get_sentiment(text)
    filler_score = audio_features.AudioFeatures.get_fillers(words)
    pitch_score = audio_features.AudioFeatures.get_pitch_variability(file_name)
    formant2_score = audio_features.AudioFeatures.get_2formant_variabilty(file_name)
    pause_score = audio_features.AudioFeatures.get_av_pause_len(audio_data)

    loneliness_index = sent_score + filler_score + pitch_score + formant2_score + pause_score

    if verbose:
        return loneliness_index, sent_score, filler_score, pitch_score, formant2_score, pause_score, text
    else:
        return loneliness_index


def get_depression_index(file_name_m4a, verbose=bool):
    file_name, audio_data, samplerate = audio_features.ProcessAudio.convert_extract_audio_data(file_name_m4a)
    text, words = audio_features.ProcessAudioprocess_audio.get_text_from_speech(file_name, audio_data)

    sent_score = audio_features.AudioFeatures.get_sentiment(text)
    filler_score = audio_features.AudioFeatures.get_fillers(words)
    pitch_score = audio_features.AudioFeatures.get_pitch_variability(file_name)
    pause_score = audio_features.AudioFeatures.get_av_pause_len(audio_data)
    mfcc2_score = audio_features.AudioFeatures.get_mfcc2(audio_data, samplerate)
    rate_of_speech = audio_features.AudioFeatures.get_speech_rate(words, audio_data, samplerate)

    depression_index = sent_score + filler_score + pitch_score + mfcc2_score + pause_score + rate_of_speech

    if verbose:
        return depression_index, sent_score, filler_score, pitch_score, mfcc2_score, pause_score, text
    else:
        return depression_index


def get_anxiety_index(file_name_m4a, verbose=bool):
    file_name, audio_data, samplerate = audio_features.ProcessAudio.convert_extract_audio_data(file_name_m4a)
    text, words = audio_features.ProcessAudioprocess_audio.get_text_from_speech(file_name, audio_data)

    f0_score = audio_features.AudioFeatures.get_f0(audio_data, samplerate) ## same as f0?
    shimmer_score = audio_features.AudioFeatures.get_shimmer(file_name)
    pause_score = audio_features.AudioFeatures.get_number_of_pauses(audio_data,samplerate)
    hnr_score = audio_features.AudioFeatures.get_hnr(file_name)

    anxiety_index = f0_score + shimmer_score + pause_score + hnr_score

    if verbose:
        return anxiety_index, f0_score, shimmer_score, pause_score, hnr_score, text
    else:
        return anxiety_index

