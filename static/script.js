
// Theme Toggle
const chartInstances = {};

// Theme Toggle
function toggleTheme() {
    document.body.classList.toggle('light-mode');

    // Update all charts
    const isLight = document.body.classList.contains('light-mode');
    const actualColor = isLight ? '#64748b' : '#94a3b8';
    const gridColor = isLight ? '#cbd5e1' : '#334155';
    const textColor = isLight ? '#0f172a' : '#f8fafc';

    for (const id in chartInstances) {
        const chart = chartInstances[id];

        // Update Actual Line Color
        if (chart.data.datasets.length > 0) {
            chart.data.datasets[0].borderColor = actualColor;
        }

        // Update Legend Color
        if (chart.options.plugins.legend) {
            chart.options.plugins.legend.labels.color = textColor;
        }

        // Update Scales
        if (chart.options.scales.y) {
            chart.options.scales.y.grid.color = gridColor;
            chart.options.scales.y.ticks.color = textColor;
        }

        chart.update();
    }
}

// Auto Load Top 10
window.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch('/api/top10');
        const tickers = await response.json();

        // Load sequentially to avoid overwhelming server CPU (fetching + training 10 models parallel is heavy)
        for (const ticker of tickers) {
            await addTicker(ticker);
        }
    } catch (e) {
        console.error("Failed to load Top 10:", e);
    }
});

async function addTicker(tickerOverride = null) {
    let ticker;
    const input = document.getElementById('tickerInput');

    if (tickerOverride) {
        ticker = tickerOverride;
    } else {
        ticker = input.value.toUpperCase().trim();
    }

    if (!ticker) return;

    const grid = document.getElementById('grid');

    // Create Card Placeholder
    const cardId = `card-${ticker}-${Date.now()}`;
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
        <div class="card-header">
            <div class="card-title">${ticker}</div>
            <div class="status" id="status-${cardId}">Loading... <div class="loader"></div></div>
        </div>
        <div class="chart-container">
            <canvas id="canvas-${cardId}"></canvas>
        </div>
    `;
    grid.prepend(card);
    if (!tickerOverride) {
        input.value = '';
    }

    try {
        const response = await fetch(`/api/predict/${ticker}`);
        const data = await response.json();

        if (response.status !== 200) {
            document.getElementById(`status-${cardId}`).innerHTML = `<span style="color: #ef4444">${data.error}</span>`;
            return;
        }

        document.getElementById(`status-${cardId}`).innerText = 'Forecast Ready';

        // Add Metrics Overlay
        const container = document.getElementById(`canvas-${cardId}`).parentElement;
        const metricsDiv = document.createElement('div');
        metricsDiv.className = 'metrics-overlay';
        metricsDiv.innerHTML = `
            MSE: ${data.metrics.mse.toFixed(2)}<br>
            MAPE: ${data.metrics.mape.toFixed(2)}%
        `;
        container.appendChild(metricsDiv);

        renderChart(cardId, data);

    } catch (e) {
        console.error(e);
        document.getElementById(`status-${cardId}`).innerText = 'Error';
    }
}

function renderChart(cardId, data) {
    const ctx = document.getElementById(`canvas-${cardId}`).getContext('2d');

    // Data Preparation
    const actual = data.backtest.actual; // Array of actual prices
    const predicted = data.backtest.predicted; // Array of predicted backtest prices
    const forecast = data.forecast; // Array of future predictions

    // Labels (indices for now)
    const totalPoints = actual.length + forecast.length;
    const labels = Array.from({ length: totalPoints }, (_, i) => i);

    // Alignment
    // Actual: [0 ... N-1]
    // Predicted Backtest: [0 ... N-1]
    // Forecast: [N ... N+M]

    // Create dataset arrays with nulls for correct overlap
    const plotActual = actual.concat(new Array(forecast.length).fill(null));
    const plotPredicted = predicted.concat(new Array(forecast.length).fill(null));

    // For forecast, we need to start where actual ended?
    // Let's connect the last point of actual to the first point of forecast visually
    const lastActual = actual[actual.length - 1];
    const plotForecast = new Array(actual.length - 1).fill(null);
    plotForecast.push(lastActual); // Connector
    plotForecast.push(...forecast);

    // Check theme for colors
    const isLight = document.body.classList.contains('light-mode');
    const actualColor = isLight ? '#64748b' : '#94a3b8';
    const gridColor = isLight ? '#cbd5e1' : '#334155';
    const textColor = isLight ? '#0f172a' : '#f8fafc';

    if (chartInstances[cardId]) {
        chartInstances[cardId].destroy();
    }

    chartInstances[cardId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Actual',
                    data: plotActual,
                    borderColor: actualColor,
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Backtest Model',
                    data: plotPredicted,
                    borderColor: '#38bdf8', // Blue
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Forecast',
                    data: plotForecast,
                    borderColor: '#4ade80', // Green
                    borderWidth: 2,
                    pointRadius: 2,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: {
                    labels: { color: textColor }
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    display: false // Sparkline feel
                },
                y: {
                    grid: {
                        color: gridColor
                    },
                    ticks: {
                        color: textColor
                    }
                }
            }
        }
    });
}
