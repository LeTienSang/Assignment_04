# Diabetes Prediction

## Chạy backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python notebook/train.py
python -m uvicorn api.main:app --reload --port 8001
```

API: `POST http://localhost:8001/predict/diabetes`  
Swagger: `http://localhost:8001/docs`

Notebook `notebook/train.py` huấn luyện baseline và đối chứng CNN 3/5 lớp (`Conv1D(16, 3)`, `Conv1D(8, 3)`, tùy chọn `MaxPool1D(2)`) với Sigmoid/BCE. Kết quả lưu tại `model/comparison.csv`; pipeline tại `model/cnn_pipeline.joblib` và trọng số PyTorch tại `model/model_3l.pt`, `model/model_5l.pt`. NumPy primitives nằm ở `common/numpy_cnn.py`; TensorFlow/Keras builder nằm ở `common/tensorflow_cnn.py`.

## Chạy mobile

```powershell
cd mobile
npm start
```

Nhấn `a` để chạy Android Emulator. App gọi API qua `http://10.0.2.2:8001`.
Khi dùng điện thoại thật, đổi `API_URL` trong `mobile/App.tsx` sang IP LAN của máy tính.
