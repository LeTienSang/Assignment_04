# Vietnam Housing Price Prediction 2024

## Chạy backend

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python notebook/train.py
python -m uvicorn api.main:app --reload --port 8002
```

API: `POST http://localhost:8002/predict/house-price`  
Swagger: `http://localhost:8002/docs`

Notebook `notebook/train.py` so sánh Random Forest, Extra Trees, Gradient Boosting và CNN 5 lớp với MSE. Kết quả lưu tại `model/comparison.csv`; pipeline tại `model/cnn_pipeline.joblib`, trọng số tại `model/model_5l.pt`, preprocessor chỉ fit trên train tại `model/preprocessor.joblib`. Các metric gồm MAE, MSE, RMSE và R2.

## Chạy mobile

```powershell
cd mobile
npm start
```

Nhấn `a` để chạy Android Emulator. App gọi API qua `http://10.0.2.2:8002`.
Khi dùng điện thoại thật, đổi `API_URL` trong `mobile/App.tsx` sang IP LAN của máy tính.
