from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

def transcribe_audio(audio_file_path):
    """Transcribe audio using PhoWhisper-tiny model."""
    try:
        processor = WhisperProcessor.from_pretrained("vinai/PhoWhisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("vinai/PhoWhisper-tiny")
        audio, sr = librosa.load(audio_file_path, sr=16000)
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
        predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        return transcription[0] if transcription else ""
    except Exception as e:
        return f"Transcription error: {str(e)}"