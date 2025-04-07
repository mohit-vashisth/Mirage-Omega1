import librosa
import noisereduce as nr
import soundfile as sf
import os

def reduce_noise_from_file(input_path, output_path=None, sr_target=16000):
    print(f"🔊 Loading: {input_path}")
    y, sr = librosa.load(input_path, sr=None)

    if sr != sr_target:
        print(f"🎯 Resampling from {sr} Hz to {sr_target} Hz")
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
        sr = sr_target

    print("🧼 Applying noise reduction...")
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_denoised.wav"

    print(f"💾 Saving cleaned audio to: {output_path}")
    sf.write(output_path, y_denoised, sr)

    return output_path

input_file = r"D:\Mirage-Omega1\ml_models\ete-synthesis\audios\mirage_1.wav"
output_file = r"D:\Mirage-Omega1\ml_models\ete-synthesis\cleaner\sample_clean.wav"

cleaned_path = reduce_noise_from_file(input_file, output_file)
print(f"Cleaned audio saved at: {cleaned_path}")