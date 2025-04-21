import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings

class AudioPipeline:
    """
    A comprehensive audio processing pipeline using TorchAudio.
    Handles device selection, audio loading, preprocessing, feature extraction, and inference.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 400,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        n_mfcc: int = 13,
        model_path: Optional[str] = None,
    ):
        """
        Initialize the audio pipeline with configurable parameters.
        
        Args:
            sample_rate: Target sample rate for audio
            n_fft: FFT size for spectrogram computation
            win_length: Window length for spectrogram computation
            hop_length: Hop length for spectrogram computation
            n_mels: Number of mel filterbanks
            n_mfcc: Number of MFCC coefficients
            model_path: Path to a pretrained model (optional)
        """
        # Check for GPU availability and set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Set audio parameters
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
        # Initialize transforms
        self._init_transforms()
        
        # Initialize model if provided
        self.model = None
        if model_path is not None:
            self.load_model(model_path)
    
    def _init_transforms(self):
        """Initialize all audio transforms."""
        # Spectrogram transform
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=2.0
        ).to(self.device)
        
        # Mel spectrogram transform
        self.melspectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        ).to(self.device)
        
        # MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'n_mels': self.n_mels,
                'hop_length': self.hop_length,
                'mel_scale': 'htk',
            }
        ).to(self.device)
        
        # Time stretch and pitch shift (for augmentation)
        self.time_stretch = torchaudio.transforms.TimeStretch(
            hop_length=self.hop_length,
            n_freq=self.n_fft // 2 + 1,
        ).to(self.device)
        
        # Resampler for handling different sample rates
        self.resampler = None  # Will be initialized when needed
    
    def load_audio(
        self, 
        file_path: str,
        normalize: bool = True,
        target_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Load audio file and perform initial preprocessing.
        
        Args:
            file_path: Path to the audio file
            normalize: Whether to normalize the audio
            target_length: Optional target length to pad or trim to
            
        Returns:
            Tuple of (waveform, sample_rate)
        """
        # Load audio file
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception as e:
            raise RuntimeError(f"Error loading audio file {file_path}: {e}")
        
        # Resample if needed
        if sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(
                    orig_freq=sr, new_freq=self.sample_rate
                ).to(self.device)
            
            waveform = self.resampler(waveform)
            sr = self.sample_rate
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Move to device
        waveform = waveform.to(self.device)
        
        # Normalize if requested
        if normalize:
            waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
        
        # Handle target length if specified
        if target_length is not None:
            if waveform.shape[1] < target_length:
                # Pad with zeros
                padding = target_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, padding))
            elif waveform.shape[1] > target_length:
                # Trim
                waveform = waveform[:, :target_length]
        
        return waveform, sr
    
    def extract_features(
        self, 
        waveform: torch.Tensor, 
        feature_type: str = 'melspectrogram',
        log_scale: bool = True,
        normalize_features: bool = True,
    ) -> torch.Tensor:
        """
        Extract audio features from waveform.
        
        Args:
            waveform: Input audio waveform tensor
            feature_type: Type of features to extract ('spectrogram', 'melspectrogram', or 'mfcc')
            log_scale: Whether to convert to log scale
            normalize_features: Whether to normalize features
            
        Returns:
            Feature tensor
        """
        # Ensure waveform is on the correct device
        waveform = waveform.to(self.device)
        
        # Extract requested feature
        if feature_type == 'spectrogram':
            features = self.spectrogram_transform(waveform)
            if log_scale:
                features = torch.log(features + 1e-8)
                
        elif feature_type == 'melspectrogram':
            features = self.melspectrogram_transform(waveform)
            if log_scale:
                features = torch.log(features + 1e-8)
                
        elif feature_type == 'mfcc':
            features = self.mfcc_transform(waveform)
            
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
        
        # Normalize if requested
        if normalize_features:
            mean = torch.mean(features, dim=(-1, -2), keepdim=True)
            std = torch.std(features, dim=(-1, -2), keepdim=True) + 1e-8
            features = (features - mean) / std
            
        return features
    
    def apply_augmentation(
        self, 
        waveform: torch.Tensor,
        augmentation_types: List[str] = ['noise', 'time_stretch', 'pitch_shift']
    ) -> torch.Tensor:
        """
        Apply audio augmentations.
        
        Args:
            waveform: Input audio waveform
            augmentation_types: List of augmentation types to apply
            
        Returns:
            Augmented waveform
        """
        # Ensure waveform is on the correct device
        waveform = waveform.to(self.device)
        augmented = waveform.clone()
        
        for aug_type in augmentation_types:
            if aug_type == 'noise':
                # Add Gaussian noise
                noise_level = 0.005 * torch.rand(1).item()
                noise = torch.randn_like(augmented) * noise_level
                augmented = augmented + noise
                
            elif aug_type == 'time_stretch':
                # Convert to spectrogram for time stretching
                spec = self.spectrogram_transform(augmented)
                # Random stretch factor between 0.8 and 1.2
                stretch_factor = 0.8 + 0.4 * torch.rand(1).item()
                spec_stretched = self.time_stretch(spec, stretch_factor)
                # Convert back to waveform using Griffin-Lim
                augmented = torchaudio.transforms.GriffinLim(
                    n_fft=self.n_fft,
                    win_length=self.win_length,
                    hop_length=self.hop_length
                ).to(self.device)(spec_stretched)
                
            elif aug_type == 'pitch_shift':
                # Simple pitch shift using resampling
                # First resample to higher/lower rate
                shift_factor = 0.8 + 0.4 * torch.rand(1).item()
                temp_resample = torchaudio.transforms.Resample(
                    orig_freq=self.sample_rate,
                    new_freq=int(self.sample_rate * shift_factor)
                ).to(self.device)
                
                # Then resample back to original rate
                restore_resample = torchaudio.transforms.Resample(
                    orig_freq=int(self.sample_rate * shift_factor),
                    new_freq=self.sample_rate
                ).to(self.device)
                
                augmented = restore_resample(temp_resample(augmented))
                
                # Trim or pad to original length
                if augmented.shape[1] < waveform.shape[1]:
                    padding = waveform.shape[1] - augmented.shape[1]
                    augmented = F.pad(augmented, (0, padding))
                else:
                    augmented = augmented[:, :waveform.shape[1]]
        
        return augmented
    
    def load_model(self, model_path: str):
        """
        Load a pretrained model.
        
        Args:
            model_path: Path to the model file
        """
        try:
            self.model = torch.jit.load(model_path).to(self.device)
            self.model.eval()
            print(f"Model loaded from {model_path}")
        except Exception as e:
            # Fallback to regular model loading if TorchScript loading fails
            try:
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                print(f"Model loaded from {model_path}")
            except Exception as e2:
                self.model = None
                warnings.warn(f"Failed to load model: {e2}")
    
    def inference(self, features: torch.Tensor) -> torch.Tensor:
        """
        Run inference on extracted features.
        
        Args:
            features: Input features tensor
            
        Returns:
            Model output tensor
        """
        if self.model is None:
            raise ValueError("No model loaded for inference")
        
        # Ensure features are on the correct device
        features = features.to(self.device)
        
        # Add batch dimension if needed
        if features.dim() == 2:
            features = features.unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(features)
            
        return output
    
    def visualize(
        self, 
        waveform: torch.Tensor = None, 
        features: torch.Tensor = None,
        feature_type: str = 'melspectrogram',
        figsize: Tuple[int, int] = (10, 8)
    ):
        """
        Visualize waveform and/or features.
        
        Args:
            waveform: Waveform tensor to visualize
            features: Features tensor to visualize
            feature_type: Type of features ('spectrogram', 'melspectrogram', or 'mfcc')
            figsize: Figure size tuple
        """
        n_plots = sum(x is not None for x in [waveform, features])
        if n_plots == 0:
            return
        
        plt.figure(figsize=figsize)
        plot_idx = 1
        
        if waveform is not None:
            plt.subplot(n_plots, 1, plot_idx)
            waveform_cpu = waveform.cpu().numpy()
            plt.plot(waveform_cpu[0])
            plt.title("Waveform")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")
            plot_idx += 1
        
        if features is not None:
            plt.subplot(n_plots, 1, plot_idx)
            features_cpu = features.cpu().numpy()
            if features_cpu.ndim > 2:
                features_cpu = features_cpu[0]  # Take first channel
                
            plt.imshow(features_cpu, aspect='auto', origin='lower')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{feature_type}")
            plt.xlabel("Time")
            
            if feature_type == 'mfcc':
                plt.ylabel("MFCC Coefficients")
            else:
                plt.ylabel("Frequency Bin")
        
        plt.tight_layout()
        plt.show()
    
    def process_file(
        self,
        file_path: str,
        feature_type: str = 'melspectrogram',
        normalize: bool = True,
        apply_augmentation: bool = False,
        run_inference: bool = False,
        visualize: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single audio file through the pipeline.
        
        Args:
            file_path: Path to the audio file
            feature_type: Type of features to extract
            normalize: Whether to normalize the audio
            apply_augmentation: Whether to apply augmentation
            run_inference: Whether to run inference (requires model)
            visualize: Whether to visualize results
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        # Load audio
        waveform, sr = self.load_audio(file_path, normalize=normalize)
        results["waveform"] = waveform
        
        # Apply augmentation if requested
        if apply_augmentation:
            waveform = self.apply_augmentation(waveform)
            results["augmented_waveform"] = waveform
        
        # Extract features
        features = self.extract_features(waveform, feature_type=feature_type)
        results["features"] = features
        
        # Run inference if requested
        if run_inference:
            if self.model is None:
                warnings.warn("Inference requested but no model loaded")
            else:
                output = self.inference(features)
                results["output"] = output
        
        # Visualize if requested
        if visualize:
            self.visualize(waveform, features, feature_type)
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = AudioPipeline(
        sample_rate=22050,
        n_fft=400,
        win_length=400,
        hop_length=160,
        n_mels=80,
        n_mfcc=13
    )
    
    # Example preprocessing
    audio_path = r"C:\Users\Admin\Documents\Mirage-Omega1\ml_models\ete_synthesis\processed_audios\mirage_1.wav"  # Replace with actual path
    
    # Process a file without visualization for speed
    try:
        results = pipeline.process_file(
            audio_path,
            feature_type='melspectrogram',
            normalize=True,
            apply_augmentation=False,
            run_inference=True,
            visualize=True
        )
        
        print(f"Waveform shape: {results['waveform'].shape}")
        print(f"Features shape: {results['features'].shape}")
        
    except FileNotFoundError:
        print(f"File {audio_path} not found. Please provide a valid file path.")
    except Exception as e:
        print(f"Error processing file: {e}")
    
    # Example batch processing function
    def process_batch(file_paths, feature_type='melspectrogram'):
        all_features = []
        for path in file_paths:
            try:
                results = pipeline.process_file(path, feature_type=feature_type)
                all_features.append(results['features'])
            except Exception as e:
                print(f"Error processing {path}: {e}")
        return all_features
    
    print("Pipeline initialized and ready to use!")