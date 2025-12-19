
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
            <div class="card-title">
                ${ticker} 
                <button class="expand-btn" onclick='openDetailedView(${JSON.stringify(ticker)})' title="Expand View">
                    <img src="/static/expand-icon.png" alt="Expand">
                </button>
            </div>
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

        // Attach data to the window or button to be accessible later.
        // Actually passing it via onclick closure is tricky if data is huge.
        // Better: store in a global map
        window.FULL_DATA_CACHE = window.FULL_DATA_CACHE || {};
        window.FULL_DATA_CACHE[ticker] = data;

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

// Global variable for detailed chart instance
let detailedChart = null;

function openDetailedView(ticker) {
    const data = window.FULL_DATA_CACHE[ticker];
    if (!data) return;

    document.getElementById('modal').style.display = 'block';
    document.getElementById('modalTitle').innerText = `${ticker} - Detailed Analysis`;

    renderDetailedChart(data);
}

function closeModal() {
    document.getElementById('modal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function (event) {
    const modal = document.getElementById('modal');
    if (event.target == modal) {
        modal.style.display = "none";
    }
}

function renderChart(cardId, data) {
    // ... Existing small chart render ...
    const ctx = document.getElementById(`canvas-${cardId}`).getContext('2d');
    const isLight = document.body.classList.contains('light-mode');

    // Prepare Data for Small Chart (Test + Forecast)
    // We only show "backtest" area + forecast for the small card
    const actual = data.backtest.actual;
    const predicted = data.backtest.predicted;
    const forecast = data.forecast;

    // ... [Reuse existing logic, but clean up references to 'data'] ...
    // Simplified Logic for replace:

    const totalPoints = actual.length + forecast.length;
    const labels = Array.from({ length: totalPoints }, (_, i) => i);

    const plotActual = actual.concat(new Array(forecast.length).fill(null));
    const plotPredicted = predicted.concat(new Array(forecast.length).fill(null));

    const lastActual = actual[actual.length - 1];
    const plotForecast = new Array(actual.length - 1).fill(null);
    plotForecast.push(lastActual);
    plotForecast.push(...forecast);

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
                    label: 'Backtest',
                    data: plotPredicted,
                    borderColor: '#38bdf8', /* Sky Blue */
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.1
                },
                {
                    label: 'Forecast',
                    data: plotForecast,
                    borderColor: '#4ade80', /* Green */
                    borderWidth: 2,
                    pointRadius: 2,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            plugins: {
                legend: { labels: { color: textColor } },
                tooltip: { mode: 'index', intersect: false }
            },
            scales: {
                x: { display: false },
                y: {
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                }
            }
        }
    });
}

function renderDetailedChart(data) {
    const ctx = document.getElementById('detailChart').getContext('2d');
    const isLight = document.body.classList.contains('light-mode');
    const textColor = isLight ? '#0f172a' : '#f8fafc';
    const gridColor = isLight ? '#cbd5e1' : '#334155';

    if (detailedChart) {
        detailedChart.destroy();
    }

    // Full History Data
    const prices = data.full_history;
    const labels = Array.from({ length: prices.length }, (_, i) => i);

    // Technicals
    const ma15 = data.technicals.mas.ma15;
    const ma30 = data.technicals.mas.ma30;
    const ma60 = data.technicals.mas.ma60;
    const ma180 = data.technicals.mas.ma180;

    // Trend Points
    const trendPoints = data.technicals.trend_points;
    const scatterData = trendPoints.map(p => ({
        x: p.index,
        y: p.value,
        type: p.type
    }));

    // Separate Peaks and Valleys for styling
    const peaks = scatterData.filter(d => d.type === 'peak' || d.type === 'peark'); // Handle type
    const valleys = scatterData.filter(d => d.type === 'valley');

    detailedChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Price',
                    data: prices,
                    borderColor: isLight ? '#0f172a' : '#f8fafc',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    order: 1
                },
                {
                    label: 'MA15',
                    data: ma15,
                    borderColor: '#f472b6', /* Pink */
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [2, 2],
                    tension: 0.2,
                    order: 2
                },
                {
                    label: 'MA30',
                    data: ma30,
                    borderColor: '#a78bfa', /* Purple */
                    borderWidth: 1,
                    pointRadius: 0,
                    borderDash: [2, 2],
                    tension: 0.2,
                    order: 3
                },
                {
                    label: 'MA60',
                    data: ma60,
                    borderColor: '#fbbf24', /* Amber */
                    borderWidth: 1,
                    pointRadius: 0,
                    tension: 0.2,
                    order: 4
                },
                {
                    label: 'MA180',
                    data: ma180,
                    borderColor: '#ef4444', /* Red */
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.2,
                    order: 5
                },
                {
                    label: 'Peaks',
                    data: peaks,
                    type: 'scatter',
                    backgroundColor: '#10b981', // Green for sell signal? Or just highlight
                    pointRadius: 6,
                    pointStyle: 'triangle',
                    order: 0
                },
                {
                    label: 'Valleys',
                    data: valleys,
                    type: 'scatter',
                    backgroundColor: '#ef4444', // Red 
                    pointRadius: 6,
                    pointStyle: 'triangle',
                    rotation: 180, // Inverted triangle
                    order: 0
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
                title: {
                    display: true,
                    text: 'Long Term Trend & Key Inflection Points',
                    color: textColor
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { display: false } // Hide dates for cleanliness or show? Hide for now as they are indices
                },
                y: {
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                }
            }
        }
    });
}
