import whisper_timestamped as whisper

AUDIO_DIR = 'fisher/'


def load():
    audio = whisper.load_audio()
