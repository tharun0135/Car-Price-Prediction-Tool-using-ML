from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import json

app = FastAPI(title="India Car Price Diagnostic HUD")

DATA_FILENAME = "india_car_dataset_700.csv"
MODEL_FILENAME = "car_price_pipeline.pkl"

NUMERIC_FEATURES = [
    "year",
    "engine_cc",
    "horsepower",
    "mileage_kmpl",
    "engine_size",
    "range_km",
]

CATEGORICAL_FEATURES = [
    "fuel_type",
    "car_type",
    "engine_type",
    "fuel_system",
    "drive_wheel",
    "transmission",
    "cylinders",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SEARCH_FIELDS = [
    "brand",
    "model",
    "fuel_type",
    "car_type",
    "engine_type",
    "fuel_system",
    "drive_wheel",
    "transmission",
]

try:
    dataset = pd.read_csv(DATA_FILENAME)
except FileNotFoundError:
    raise RuntimeError(f"Dataset file not found: {DATA_FILENAME}")

try:
    model = joblib.load(MODEL_FILENAME)
except Exception:
    model = None


class PredictRequest(BaseModel):
    year: int
    engine_cc: int
    horsepower: int
    mileage_kmpl: float
    engine_size: float
    range_km: int
    fuel_type: str
    car_type: str
    engine_type: str
    fuel_system: str
    drive_wheel: str
    transmission: str
    cylinders: int
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None


class SearchFilters(BaseModel):
    fuel_type: Optional[str] = None
    car_type: Optional[str] = None
    engine_type: Optional[str] = None
    fuel_system: Optional[str] = None
    drive_wheel: Optional[str] = None
    transmission: Optional[str] = None
    min_price_lakh: Optional[float] = None
    max_price_lakh: Optional[float] = None
    min_horsepower: Optional[int] = None
    max_horsepower: Optional[int] = None
    min_mileage_kmpl: Optional[float] = None
    max_mileage_kmpl: Optional[float] = None


def _build_select_options(values):
    return sorted([str(x) for x in set(values) if pd.notna(x)])


@app.get("/", response_class=HTMLResponse)
def read_root():
    numeric_ranges = {}
    for feature in NUMERIC_FEATURES:
        col = dataset[feature].dropna()
        numeric_ranges[feature] = {
            "min": int(col.min()),
            "max": int(col.max()),
            "value": int(col.median()),
        }

    option_choices = {
        "fuel_type": _build_select_options(dataset["fuel_type"]),
        "car_type": _build_select_options(dataset["car_type"]),
        "engine_type": _build_select_options(dataset["engine_type"]),
        "fuel_system": _build_select_options(dataset["fuel_system"]),
        "drive_wheel": _build_select_options(dataset["drive_wheel"]),
        "transmission": _build_select_options(dataset["transmission"]),
        "cylinders": _build_select_options(dataset["cylinders"]),
    }

    chart_features = ["year", "engine_cc", "horsepower", "mileage_kmpl"]
    chart_data = {feature: {} for feature in chart_features}
    
    # Render line charts for numeric features
    for feature in chart_features:
        series = dataset.dropna(subset=[feature, "price_lakh"])
        bins = pd.cut(series[feature], bins=10)
        # Added observed=False to prevent FutureWarnings in pandas 2.1+
        avg_prices = series.groupby(bins, observed=False)["price_lakh"].mean().fillna(0)
        labels = [f"{interval.left:.0f}-{interval.right:.0f}" for interval in avg_prices.index]
        chart_data[feature] = {"labels": labels, "data": avg_prices.tolist()}

    # Render bar chart specific data for fuel_type (Fixing the logic gap)
    fuel_series = dataset.dropna(subset=["fuel_type", "price_lakh"])
    fuel_prices = fuel_series.groupby("fuel_type", observed=False)["price_lakh"].mean().fillna(0)
    chart_data["fuel_type"] = {
        "labels": fuel_prices.index.astype(str).tolist(), 
        "data": fuel_prices.tolist()
    }

    price_min = float(dataset["price_lakh"].min())
    price_max = float(dataset["price_lakh"].max())

    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>India Car Price Diagnostic HUD</title>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: "Courier New", monospace;
            background: radial-gradient(ellipse at center, #05121a 0%, #000000 100%);
            color: #d6faff;
            overflow: hidden;
            position: relative;
        }

        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background:
                radial-gradient(circle at 20% 80%, rgba(0, 255, 255, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255, 0, 255, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(255, 255, 0, 0.02) 0%, transparent 50%);
            pointer-events: none;
            z-index: -1;
        }

        .page-wrapper {
            display: grid;
            grid-template-columns: 420px 1fr;
            min-height: 100vh;
            position: relative;
            z-index: 1;
        }

        .panel {
            padding: 32px;
            background: linear-gradient(180deg, rgba(0,16,28,0.98), rgba(0,30,53,0.98));
            border-right: 2px solid rgba(102, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            position: relative;
            overflow-y: auto;
            max-height: 100vh;
        }

        .panel::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.3), transparent);
        }

        .panel h1 {
            font-size: 28px;
            margin-bottom: 16px;
            color: #72f1ff;
            letter-spacing: 2px;
            text-shadow: 0 0 15px rgba(114, 241, 255, 0.6);
        }

        .panel p {
            margin-bottom: 28px;
            line-height: 1.6;
            color: #9fdfff;
            font-size: 14px;
        }

        .control-group {
            margin-bottom: 24px;
            position: relative;
        }

        .control-group label {
            display: block;
            margin-bottom: 10px;
            font-size: 12px;
            text-transform: uppercase;
            color: #7ed2ff;
            letter-spacing: 1px;
            font-weight: 600;
        }

        select, input[type=range], input[type=number] {
            width: 100%;
            border: none;
            border-radius: 8px;
            padding: 12px 14px;
            background: rgba(255,255,255,0.05);
            color: #eefcff;
            outline: none;
            box-shadow: inset 0 0 0 1px rgba(93, 216, 255, 0.2);
            transition: all 0.3s ease;
            font-family: inherit;
            font-size: 14px;
        }

        select:focus, input[type=range]:focus, input[type=number]:focus {
            box-shadow: inset 0 0 0 2px rgba(0, 255, 255, 0.6), 0 0 20px rgba(0, 255, 255, 0.2);
            background: rgba(255,255,255,0.08);
        }

        select:hover, input[type=range]:hover, input[type=number]:hover {
            background: rgba(255,255,255,0.08);
            transform: translateY(-1px);
        }

        .slider-container {
            position: relative;
            margin-top: 8px;
        }

        .slider-track {
            position: absolute;
            top: 50%;
            left: 0;
            right: 0;
            height: 4px;
            background: rgba(128, 128, 128, 0.3);
            border-radius: 2px;
            transform: translateY(-50%);
        }

        .slider-fill {
            position: absolute;
            top: 0;
            left: 0;
            height: 100%;
            background: linear-gradient(90deg, #00ffff, #0080ff);
            border-radius: 2px;
            transition: width 0.1s ease;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        }

        .slider-handle {
            position: absolute;
            top: 50%;
            width: 18px;
            height: 18px;
            background: #00ffff;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            cursor: pointer;
            box-shadow: 0 0 12px rgba(0, 255, 255, 0.8);
            transition: all 0.2s ease;
        }

        .slider-handle:hover {
            transform: translate(-50%, -50%) scale(1.2);
            box-shadow: 0 0 20px rgba(0, 255, 255, 1);
        }

        .output-card {
            margin-top: 32px;
            padding: 28px;
            border-radius: 16px;
            background: rgba(2, 24, 38, 0.95);
            border: 2px solid rgba(102, 255, 255, 0.2);
            position: relative;
            overflow: hidden;
        }

        .output-card::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, transparent, rgba(0, 255, 255, 0.08), transparent);
            border-radius: 16px;
            pointer-events: none;
            opacity: 0.5;
        }

        .output-value {
            font-size: 52px;
            font-weight: 700;
            color: #ffff7f;
            line-height: 1;
            text-shadow: 0 0 20px rgba(255, 255, 127, 0.8);
            transition: all 0.5s ease;
            position: relative;
        }

        .output-value.updating {
            animation: pricePulse 0.6s ease-in-out;
        }

        @keyframes pricePulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.02); }
        }

        .output-subtext {
            margin-top: 12px;
            color: #a4f3ff;
            font-size: 13px;
        }

        .chart-panel {
            padding: 32px;
            overflow-y: auto;
            background: rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
        }

        .chart-block {
            margin-bottom: 24px;
            padding: 20px;
            border-radius: 16px;
            background: rgba(2, 24, 38, 0.9);
            border: 1px solid rgba(102, 255, 255, 0.15);
            transition: all 0.3s ease;
            max-height: 300px;
            overflow: hidden;
        }

        .chart-block h2 {
            font-size: 16px !important;
            margin-bottom: 12px !important;
        }

        .chart-block:hover {
            border-color: rgba(102, 255, 255, 0.3);
            box-shadow: 0 0 30px rgba(102, 255, 255, 0.1);
            transform: translateY(-2px);
        }

        canvas {
            width: 100% !important;
            max-width: 100%;
            border-radius: 8px;
            max-height: 180px !important;
        }

        .dashboard-buttons {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-top: 20px;
        }

        .dashboard-buttons button {
            flex: 1;
            background: linear-gradient(135deg, rgba(0, 128, 255, 0.8), rgba(0, 255, 255, 0.8));
            color: #ffffff;
            border: 2px solid rgba(0, 255, 255, 0.3);
            padding: 14px 20px;
            border-radius: 12px;
            cursor: pointer;
            font-family: inherit;
            font-size: 14px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        .dashboard-buttons button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
            transition: left 0.5s ease;
        }

        .dashboard-buttons button:hover::before {
            left: 100%;
        }

        .dashboard-buttons button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(0, 255, 255, 0.3);
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.9), rgba(0, 128, 255, 0.9));
        }

        .dashboard-buttons button:active {
            transform: translateY(0);
        }

        .dashboard-buttons button.loading {
            pointer-events: none;
            opacity: 0.7;
        }

        .dashboard-buttons button.loading::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-top: 2px solid #ffffff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            right: 15px;
            top: 50%;
            transform: translateY(-50%);
        }

        @keyframes spin {
            0% { transform: translateY(-50%) rotate(0deg); }
            100% { transform: translateY(-50%) rotate(360deg); }
        }

        .status-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 10px 16px;
            border-radius: 20px;
            background: rgba(0, 255, 0, 0.2);
            border: 1px solid rgba(0, 255, 0, 0.5);
            color: #00ff00;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 1000;
        }

        .status-indicator.active {
            opacity: 1;
        }

        .particles {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }

        .particle {
            position: absolute;
            width: 1px;
            height: 1px;
            background: rgba(0, 255, 255, 0.3);
            border-radius: 50%;
            animation: float 12s linear infinite;
        }

        .particle:nth-child(2n) {
            background: rgba(255, 0, 255, 0.2);
            animation-duration: 14s;
        }

        .particle:nth-child(3n) {
            background: rgba(255, 255, 0, 0.15);
            animation-duration: 16s;
        }

        @keyframes float {
            0% {
                transform: translateY(100vh);
                opacity: 0;
            }
            10% {
                opacity: 0.3;
            }
            90% {
                opacity: 0.3;
            }
            100% {
                transform: translateY(-100vh);
                opacity: 0;
            }
        }

        @media (max-width: 1024px) {
            .page-wrapper {
                grid-template-columns: 1fr;
            }
            .panel {
                border-right: none;
                border-bottom: 2px solid rgba(102, 255, 255, 0.2);
                max-height: 50vh;
            }
        }

        @media (max-width: 768px) {
            .panel h1 {
                font-size: 24px;
            }
            .output-value {
                font-size: 42px;
            }
            .dashboard-buttons {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="page-wrapper">
        <section class="panel">
            <h1>India Car Price Diagnostic HUD</h1>
            <p>Choose the car specification and budget details below. The diagnostic engine returns an estimated price in lakhs, plus nearby market comparables.</p>
            <div id="inputs"></div>
            <div class="output-card">
                <div class="output-value" id="predicted-price">--</div>
                <div class="output-subtext">Estimated car price in lakhs</div>
                <div class="output-subtext" id="suggestion-text"></div>
            </div>
        </section>
        <section class="chart-panel">
            <div class="chart-block">
                <h2 style="margin-top:0; color:#72f1ff;">Price Trend by Feature</h2>
                <canvas id="trend-chart"></canvas>
            </div>
            <div class="chart-block">
                <h2 style="margin-top:0; color:#72f1ff;">Fuel Type Pricing Breakdown</h2>
                <canvas id="fuel-chart"></canvas>
            </div>
            <div class="dashboard-buttons">
                <button id="predict-btn">Run Diagnostic</button>
                <button id="clear-btn">Reset Inputs</button>
            </div>
        </section>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script>
        const numericRanges = __NUMERIC_RANGES__;
        const options = __OPTION_CHOICES__;
        const chartData = __CHART_DATA__;
        const priceRange = {"min": __PRICE_MIN__, "max": __PRICE_MAX__};

        // Global state
        let isUpdating = false;
        let lastPrediction = null;
        let autoSaveTimer = null;

        // Create particle system
        function createParticles() {
            const particlesContainer = document.createElement('div');
            particlesContainer.className = 'particles';
            document.body.appendChild(particlesContainer);

            for (let i = 0; i < 12; i++) {
                const particle = document.createElement('div');
                particle.className = 'particle';
                particle.style.left = Math.random() * 100 + '%';
                particle.style.animationDelay = Math.random() * 10 + 's';
                particlesContainer.appendChild(particle);
            }
        }

        // Enhanced slider with visual feedback
        function createEnhancedSlider(key, config) {
            const container = document.createElement('div');
            container.className = 'control-group';

            const label = document.createElement('label');
            label.htmlFor = key;
            label.textContent = key.replace('_', ' ').toUpperCase();
            container.appendChild(label);

            const sliderContainer = document.createElement('div');
            sliderContainer.className = 'slider-container';

            const track = document.createElement('div');
            track.className = 'slider-track';
            sliderContainer.appendChild(track);

            const fill = document.createElement('div');
            fill.className = 'slider-fill';
            sliderContainer.appendChild(fill);

            const rangeInput = document.createElement('input');
            rangeInput.id = key;
            rangeInput.type = 'range';
            rangeInput.min = config.min;
            rangeInput.max = config.max;
            rangeInput.value = config.value;
            rangeInput.step = ['mileage_kmpl', 'engine_size'].includes(key) ? 0.1 : 1;
            sliderContainer.appendChild(rangeInput);

            const handle = document.createElement('div');
            handle.className = 'slider-handle';
            sliderContainer.appendChild(handle);

            const numberInput = document.createElement('input');
            numberInput.id = `${key}-value`;
            numberInput.type = 'number';
            numberInput.value = config.value;
            numberInput.min = config.min;
            numberInput.max = config.max;
            numberInput.step = ['mileage_kmpl', 'engine_size'].includes(key) ? 0.1 : 1;

            container.appendChild(sliderContainer);
            container.appendChild(numberInput);

            return container;
        }

        function createSelectControl(key, values) {
            const container = document.createElement('div');
            container.className = 'control-group';

            const label = document.createElement('label');
            label.htmlFor = key;
            label.textContent = key.replace('_', ' ').toUpperCase();
            container.appendChild(label);

            const select = document.createElement('select');
            select.id = key;
            values.forEach(value => {
                const option = document.createElement('option');
                option.value = value;
                option.textContent = value;
                select.appendChild(option);
            });
            container.appendChild(select);

            return container;
        }

        function renderControls() {
            // FIX: explicitly grab inputsContainer to fix ReferenceError
            const inputsContainer = document.getElementById('inputs');
            inputsContainer.innerHTML = '';
            
            Object.entries(numericRanges).forEach(([key, config]) => {
                inputsContainer.appendChild(createEnhancedSlider(key, config));
            });
            Object.entries(options).forEach(([key, values]) => {
                inputsContainer.appendChild(createSelectControl(key, values));
            });

            // Add budget controls
            const budgetMinGroup = document.createElement('div');
            budgetMinGroup.className = 'control-group';
            budgetMinGroup.innerHTML = `
                <label for="budget_min">BUDGET MIN (Lakh)</label>
                <input id="budget_min" type="number" step="0.1" min="${priceRange.min}" max="${priceRange.max}" value="${priceRange.min}" />
            `;
            inputsContainer.appendChild(budgetMinGroup);

            const budgetMaxGroup = document.createElement('div');
            budgetMaxGroup.className = 'control-group';
            budgetMaxGroup.innerHTML = `
                <label for="budget_max">BUDGET MAX (Lakh)</label>
                <input id="budget_max" type="number" step="0.1" min="${priceRange.min}" max="${priceRange.max}" value="${priceRange.max}" />
            `;
            inputsContainer.appendChild(budgetMaxGroup);
        }

        function updateSliderVisuals() {
            Object.keys(numericRanges).forEach(key => {
                const rangeEl = document.getElementById(key);
                const fill = rangeEl.parentElement.querySelector('.slider-fill');
                const handle = rangeEl.parentElement.querySelector('.slider-handle');

                if (rangeEl && fill && handle) {
                    const percentage = ((rangeEl.value - rangeEl.min) / (rangeEl.max - rangeEl.min)) * 100;
                    fill.style.width = percentage + '%';
                    handle.style.left = percentage + '%';
                }
            });
        }

        function syncRangeWithInput(rangeEl, numberEl) {
            rangeEl.addEventListener('input', () => {
                numberEl.value = rangeEl.value;
                updateSliderVisuals();
                scheduleAutoPrediction();
            });
            numberEl.addEventListener('input', () => {
                rangeEl.value = numberEl.value;
                updateSliderVisuals();
                scheduleAutoPrediction();
            });
            numberEl.addEventListener('blur', () => {
                if (numberEl.value < rangeEl.min) numberEl.value = rangeEl.min;
                if (numberEl.value > rangeEl.max) numberEl.value = rangeEl.max;
                rangeEl.value = numberEl.value;
                updateSliderVisuals();
            });
        }

        function initializeControls() {
            renderControls();
            Object.keys(numericRanges).forEach(key => {
                const rangeEl = document.getElementById(key);
                const numberEl = document.getElementById(`${key}-value`);
                syncRangeWithInput(rangeEl, numberEl);
            });
            updateSliderVisuals();

            // Add change listeners for select elements
            Object.keys(options).forEach(key => {
                document.getElementById(key).addEventListener('change', scheduleAutoPrediction);
            });

            // Add change listeners for budget inputs
            document.getElementById('budget_min').addEventListener('input', scheduleAutoPrediction);
            document.getElementById('budget_max').addEventListener('input', scheduleAutoPrediction);
        }

        function scheduleAutoPrediction() {
            if (autoSaveTimer) clearTimeout(autoSaveTimer);
            autoSaveTimer = setTimeout(() => {
                if (!isUpdating) runPrediction(true);
            }, 800); // Debounce for 800ms
        }

        function getFormData() {
            const payload = {};
            Object.keys(numericRanges).forEach(key => {
                const value = document.getElementById(key).value;
                payload[key] = ['mileage_kmpl', 'engine_size'].includes(key) ? parseFloat(value) : parseInt(value, 10);
            });
            Object.keys(options).forEach(key => {
                const value = document.getElementById(key).value;
                payload[key] = key === 'cylinders' ? parseInt(value, 10) : value;
            });
            payload.budget_min = parseFloat(document.getElementById('budget_min').value);
            payload.budget_max = parseFloat(document.getElementById('budget_max').value);
            return payload;
        }

        async function runPrediction(auto = false) {
            if (isUpdating) return;

            isUpdating = true;
            const predictBtn = document.getElementById('predict-btn');
            const priceEl = document.getElementById('predicted-price');
            const statusEl = document.querySelector('.status-indicator') || createStatusIndicator();

            if (!auto) {
                predictBtn.classList.add('loading');
                predictBtn.textContent = 'Analyzing...';
            }

            priceEl.classList.add('updating');
            statusEl.textContent = 'Processing data...';
            statusEl.classList.add('active');

            try {
                const payload = getFormData();
                const response = await fetch('/predict_and_suggest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Prediction failed');
                }

                const data = await response.json();
                lastPrediction = data;

                // Animate price update
                setTimeout(() => {
                    priceEl.textContent = `${data.predicted_price.toFixed(2)} L`;
                    priceEl.classList.remove('updating');
                }, 300);

                document.getElementById('suggestion-text').textContent =
                    data.suggestions.length ?
                    `Closest matches: ${data.suggestions.map(item => `${item.brand} ${item.model} (${item.price_lakh} L)`).slice(0,3).join(', ')}` :
                    'No comparables found in the selected budget range.';

                statusEl.textContent = 'Analysis complete';
                setTimeout(() => statusEl.classList.remove('active'), 2000);

            } catch (err) {
                priceEl.textContent = 'Error';
                priceEl.classList.remove('updating');
                document.getElementById('suggestion-text').textContent = err.message;
                statusEl.textContent = 'Error occurred';
                statusEl.style.background = 'rgba(255, 0, 0, 0.2)';
                statusEl.style.borderColor = 'rgba(255, 0, 0, 0.5)';
                statusEl.style.color = '#ff0000';
                setTimeout(() => {
                    statusEl.classList.remove('active');
                    statusEl.style.background = '';
                    statusEl.style.borderColor = '';
                    statusEl.style.color = '';
                }, 3000);
            } finally {
                isUpdating = false;
                if (!auto) {
                    predictBtn.classList.remove('loading');
                    predictBtn.textContent = 'Run Diagnostic';
                }
            }
        }

        function createStatusIndicator() {
            const statusEl = document.createElement('div');
            statusEl.className = 'status-indicator';
            document.body.appendChild(statusEl);
            return statusEl;
        }

        function clearInputs() {
            initializeControls();
            document.getElementById('predicted-price').textContent = '--';
            document.getElementById('suggestion-text').textContent = '';
            lastPrediction = null;
            if (autoSaveTimer) clearTimeout(autoSaveTimer);
        }

        function createChart() {
            const ctx1 = document.getElementById('trend-chart').getContext('2d');
            new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: chartData.year.labels,
                    datasets: [
                        {
                            label: 'Year vs Price',
                            data: chartData.year.data,
                            borderColor: '#6bf5ff',
                            backgroundColor: 'rgba(107, 245, 255, 0.16)',
                            fill: true,
                            tension: 0.3,
                            pointBackgroundColor: '#6bf5ff',
                            pointBorderColor: '#ffffff',
                            pointHoverBackgroundColor: '#ffffff',
                            pointHoverBorderColor: '#6bf5ff'
                        },
                    ],
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 16, 28, 0.9)',
                            titleColor: '#72f1ff',
                            bodyColor: '#d6faff',
                            borderColor: '#6bf5ff',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            ticks: { color: '#d6faff' },
                            grid: { color: 'rgba(102, 255, 255, 0.1)' }
                        },
                        x: {
                            ticks: { color: '#d6faff' },
                            grid: { color: 'rgba(102, 255, 255, 0.1)' }
                        }
                    },
                    animation: {
                        duration: 1000,
                        easing: 'easeInOutQuart'
                    }
                },
            });

            // FIX: Using correct chartData mapping for fuel types
            const ctx2 = document.getElementById('fuel-chart').getContext('2d');
            new Chart(ctx2, {
                type: 'bar',
                data: {
                    labels: chartData.fuel_type.labels,
                    datasets: [
                        {
                            label: 'Fuel Type vs Price',
                            data: chartData.fuel_type.data,
                            backgroundColor: 'rgba(114, 241, 255, 0.55)',
                            borderColor: '#72f1ff',
                            borderWidth: 1,
                            hoverBackgroundColor: 'rgba(114, 241, 255, 0.8)',
                            hoverBorderColor: '#ffffff'
                        },
                    ],
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(0, 16, 28, 0.9)',
                            titleColor: '#72f1ff',
                            bodyColor: '#d6faff',
                            borderColor: '#6bf5ff',
                            borderWidth: 1
                        }
                    },
                    scales: {
                        y: { ticks: { color: '#d6faff' }, grid: { color: 'rgba(102, 255, 255, 0.1)' } },
                        x: { ticks: { color: '#d6faff' }, grid: { color: 'rgba(102, 255, 255, 0.1)' } }
                    },
                    animation: {
                        duration: 1000,
                        easing: 'easeInOutQuart'
                    }
                },
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case 'Enter':
                        e.preventDefault();
                        runPrediction();
                        break;
                    case 'r':
                        e.preventDefault();
                        clearInputs();
                        break;
                }
            }
        });

        // Initialize everything
        document.addEventListener('DOMContentLoaded', () => {
            createParticles();
            initializeControls();
            createChart();

            // Add event listeners
            document.getElementById('predict-btn').addEventListener('click', () => runPrediction(false));
            document.getElementById('clear-btn').addEventListener('click', clearInputs);

            // Add visual feedback for inputs
            document.querySelectorAll('input, select').forEach(el => {
                el.addEventListener('focus', () => {
                    el.parentElement.style.transform = 'translateY(-2px)';
                });
                el.addEventListener('blur', () => {
                    el.parentElement.style.transform = '';
                });
            });
        });
    </script>
</body>
</html>
"""
    html = html.replace('__NUMERIC_RANGES__', json.dumps(numeric_ranges))
    html = html.replace('__OPTION_CHOICES__', json.dumps(option_choices))
    html = html.replace('__CHART_DATA__', json.dumps(chart_data))
    html = html.replace('__PRICE_MIN__', str(price_min))
    html = html.replace('__PRICE_MAX__', str(price_max))
    return html


@app.post('/predict_and_suggest')
def predict_and_suggest(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    # 1. Get the predicted price
    input_data = pd.DataFrame([request.model_dump()])[MODEL_FEATURES]
    prediction = float(model.predict(input_data)[0])

    # 2. Find market comparables
    suggestions = dataset.copy()
    
    # Filter by budget bounds
    if request.budget_min is not None:
        suggestions = suggestions[suggestions['price_lakh'] >= request.budget_min]
    if request.budget_max is not None:
        suggestions = suggestions[suggestions['price_lakh'] <= request.budget_max]

    # FIX: Calculate the absolute difference from the predicted price
    suggestions['price_diff'] = (suggestions['price_lakh'] - prediction).abs()
    
    # Optional but recommended: Drop duplicate models so suggestions are varied
    suggestions = suggestions.drop_duplicates(subset=['brand', 'model', 'price_lakh'])
    
    # Sort by the closest price difference (ascending)
    suggestions = suggestions.sort_values('price_diff')
    
    # Grab the top 6 closest matches
    top_suggestions = suggestions.head(6)[['brand', 'model', 'price_lakh', 'fuel_type', 'car_type', 'transmission']]
    
    return {
        'predicted_price': prediction,
        'suggestions': top_suggestions.to_dict(orient='records'),
    }

@app.post('/search')
def search_cars(filters: SearchFilters):
    results = dataset.copy()
    if filters.fuel_type:
        results = results[results['fuel_type'] == filters.fuel_type]
    if filters.car_type:
        results = results[results['car_type'] == filters.car_type]
    if filters.engine_type:
        results = results[results['engine_type'] == filters.engine_type]
    if filters.fuel_system:
        results = results[results['fuel_system'] == filters.fuel_system]
    if filters.drive_wheel:
        results = results[results['drive_wheel'] == filters.drive_wheel]
    if filters.transmission:
        results = results[results['transmission'] == filters.transmission]
    if filters.min_price_lakh is not None:
        results = results[results['price_lakh'] >= filters.min_price_lakh]
    if filters.max_price_lakh is not None:
        results = results[results['price_lakh'] <= filters.max_price_lakh]
    if filters.min_horsepower is not None:
        results = results[results['horsepower'] >= filters.min_horsepower]
    if filters.max_horsepower is not None:
        results = results[results['horsepower'] <= filters.max_horsepower]
    if filters.min_mileage_kmpl is not None:
        results = results[results['mileage_kmpl'] >= filters.min_mileage_kmpl]
    if filters.max_mileage_kmpl is not None:
        results = results[results['mileage_kmpl'] <= filters.max_mileage_kmpl]

    return results.head(20).to_dict(orient='records')