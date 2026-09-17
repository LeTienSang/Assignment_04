# Customer Behavior Intelligence

End-to-end customer purchase propensity demo with a Python ML service and React dashboard.

## Run the ML service

From `customer_behavior/`:

```powershell
python -m pip install -r requirements.txt
python notebook/train.py
python -m uvicorn api.main:app --reload --port 8000
```

The training command creates `data/ecom_data.csv`, `model/cnn_pipeline.joblib`, `model/model_5l.pt`, and `model/comparison.csv`. The CNN uses a six-feature sequence plus a deterministic 100-dimensional embedding and reports accuracy, precision, recall, and F1. The existing response contract remains unchanged.

## Run the dashboard

In a second terminal:

```powershell
cd web
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally `http://localhost:5173`.

## Run the mobile app

The React Native app lives in `mobile/` and reuses the same FastAPI endpoint.

```powershell
cd mobile
npm start
```

Then press `a` for an Android emulator, scan the QR code with Expo Go on a
physical device, or run `npm run web` to preview the mobile layout in a browser.

For a physical device, change `API_URL` in `mobile/App.tsx` from
`127.0.0.1` to the computer's local network IP, for example
`http://192.168.1.10:8000/predict`. For the Android emulator, use
`http://10.0.2.2:8000/predict`.
