import whisper

model = whisper.load_model("base")
result = model.transcribe("response.mp3")  # Thử với file audio có sẵn
print(result["text"])
