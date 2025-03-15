import sounddevice as sd

print("Đang kiểm tra micro... Hãy nói gì đó trong 5 giây!")
fs = 16000  # Tần số lấy mẫu
duration = 5  # Thời gian ghi âm (giây)
recording = sd.rec(int(fs * duration), samplerate=fs, channels=1, dtype='int16')
sd.wait()  # Chờ ghi âm xong
print("Ghi âm xong!")

if recording.any():
    print("✅ Micro hoạt động!")
else:
    print("❌ Không thu được âm thanh, kiểm tra lại micro.")
