import os
import librosa
import soundfile as sf

def convert_audio_files(folder_path, target_sr=16000):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp3')]
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        
        try:
            # Load and resample the audio
            audio, sr = librosa.load(file_path, sr=target_sr, mono=True)

            # Save as .wav
            new_file = os.path.splitext(file)[0] + '.wav'
            new_file_path = os.path.join(folder_path, new_file)
            sf.write(new_file_path, audio, target_sr)

            # Delete original .mp3 file
            os.remove(file_path)

            print(f"✅ Converted and deleted: {file}")
        except Exception as e:
            print(f"❌ Error processing {file}: {e}")

folder = r"D:\Mirage-Omega1\clips"
convert_audio_files(folder)