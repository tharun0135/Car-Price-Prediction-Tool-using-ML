# Car Price Prediction Dashboard

A modern web application for predicting car prices using machine learning and exploring market trends through interactive dashboards.

## Features

- **Price Prediction**: Input car specifications to get accurate price estimates
- **Personalized Suggestions**: Get car recommendations within your budget range (±15% of predicted price)
- **Interactive Dashboard**: Visualize how different features affect car prices with line charts
- **Modern UI**: Clean, responsive design with gradient backgrounds and smooth animations

## Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **ML**: Scikit-learn with Random Forest
- **Charts**: Chart.js for data visualization
- **Deployment**: Ready for cloud platforms (Heroku, Railway, etc.)

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Train the model (optional, pre-trained model included):
   ```bash
   python train_model.py
   ```
4. Run the application:
   ```bash
   uvicorn app:app --reload
   ```
5. Open http://127.0.0.1:8000 in your browser

## Deployment

### Heroku
1. Create a Heroku app
2. Set buildpack to Python
3. Deploy with git push

### Railway
1. Connect GitHub repo
2. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### Other Platforms
- Ensure `uvicorn app:app --host 0.0.0.0 --port $PORT` as start command
- Add `gunicorn` for production if needed

## API Endpoints

- `GET /`: Main dashboard with prediction form and charts
- `POST /predict_and_suggest`: Predict price and get suggestions
- `POST /search`: Search cars with filters
- `GET /docs`: Interactive API documentation

## Model Features

The model uses these key features for prediction:
- Engine size (cc)
- Horsepower
- Highway MPG
- City MPG
- Fuel system
- Fuel type
- Cylinder number

## Dashboard Charts

- Average price by engine size
- Average price by horsepower
- Average price by highway MPG
- Average price by city MPG

## License

MIT License