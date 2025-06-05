import torch.serialization
from TTS.api import TTS
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.config.shared_configs import BaseDatasetConfig  # NEW

# Allow all custom configs in safe list
torch.serialization.add_safe_globals([
    XttsAudioConfig,
    XttsConfig,
    BaseDatasetConfig
])

# Model load karo
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")  # or "cuda" if you have GPU

reference_audio_path = r"C:\Users\Admin\Documents\Mirage-Omega1\output.wav"
text = "oye mote dekh kardi voice clone maine."

tts.tts_to_file(
    text=text,
    file_path="cloned_output.wav",
    speaker_wav=reference_audio_path,
    language="hi"
)
