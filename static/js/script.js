
const form = document.getElementById("predictionForm");

const sunshine = document.getElementById("sunshine");
const temperature = document.getElementById("temperature");
const wind = document.getElementById("wind");
const weather = document.getElementById("weather");

const sunshineValue = document.getElementById("sunshineValue");
const temperatureValue = document.getElementById("temperatureValue");
const windValue = document.getElementById("windValue");

const predictBtn = document.getElementById("predictBtn");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");

// Update slider values
sunshine.addEventListener("input", () => {
    sunshineValue.textContent = sunshine.value;
});

temperature.addEventListener("input", () => {
    temperatureValue.textContent = temperature.value;
});

wind.addEventListener("input", () => {
    windValue.textContent = wind.value;
});

// Weather selection sets a suggested sunlight level
const weatherPresets = {
    "Sunny": 75,
    "Partly Cloudy": 50,
    "Cloudy": 25,
    "Rainy": 10
};

weather.addEventListener("change", () => {
    sunshine.value = weatherPresets[weather.value];
    sunshineValue.textContent = sunshine.value;
});

// Send values to Flask
form.addEventListener("submit", async (event) => {
    event.preventDefault();

    resultBox.style.display = "none";
    errorBox.style.display = "none";

    predictBtn.disabled = true;
    predictBtn.textContent = "Predicting...";

    const data = {
        weather: weather.value,
        sunshine_percent: Number(sunshine.value),
        temperature: Number(temperature.value),
        wind_speed: Number(wind.value)
    };

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok || !result.success) {
            throw new Error(result.error || "Prediction failed.");
        }

        document.getElementById("predictionValue").textContent =
            result.prediction.toLocaleString("en-IN") + " kWh";

        resultBox.style.display = "block";

    } catch (error) {
        errorBox.textContent = error.message;
        errorBox.style.display = "block";

    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "☀ Predict Solar Production";
    }
});
