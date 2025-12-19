
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
let currentTicker = null; // Track current ticker for detailed view

function openDetailedView(ticker) {
    currentTicker = ticker;
    const data = window.FULL_DATA_CACHE[ticker];
    if (!data) return;

    document.getElementById('modal').style.display = 'block';
    document.getElementById('modalTitle').innerText = `${ticker} - Detailed Analysis`;

    // Default to Daily View (All) or remember last? Default All.
    renderDetailedChart(data);
}

// ... closeModal ...

function renderIntradayChart(data, range) {
    const ctx = document.getElementById('detailChart').getContext('2d');
    const isLight = document.body.classList.contains('light-mode');
    const textColor = isLight ? '#0f172a' : '#f8fafc';
    const gridColor = isLight ? '#cbd5e1' : '#334155';

    if (detailedChart) {
        detailedChart.destroy();
    }

    if (!data.intraday) {
        // Fallback to daily if no intraday data
        renderDetailedChart(data);
        updateChartRange(range); // Apply zoom on daily
        return;
    }

    const { dates, prices } = data.intraday;

    // Slice data based on range
    let sliceLen = prices.length;
    // dates are 'YYYY-MM-DD HH:MM'
    // 1h interval.
    // 1d = 7 hours
    // 3d = 21 hours
    // 7d = 49 hours
    // 1m = all (since we fetched 1mo)

    switch (range) {
        case '1d': sliceLen = 7; break; // Approx
        case '3d': sliceLen = 21; break;
        case '7d': sliceLen = 49; break;
        case '1m': sliceLen = prices.length; break;
    }

    if (sliceLen > prices.length) sliceLen = prices.length;

    const startIdx = prices.length - sliceLen;
    const slicedDates = dates.slice(startIdx);
    const slicedPrices = prices.slice(startIdx);

    detailedChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: slicedDates,
            datasets: [
                {
                    label: 'Intraday Price',
                    data: slicedPrices,
                    borderColor: isLight ? '#0f172a' : '#f8fafc',
                    borderWidth: 2,
                    pointRadius: 2, // Show points for detail
                    tension: 0.2,
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
                legend: { labels: { color: textColor } },
                title: {
                    display: true,
                    text: `Intraday High-Res View (${range})`,
                    color: textColor
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: { display: false },
                    ticks: {
                        color: textColor,
                        maxTicksLimit: 8
                    }
                },
                y: {
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                }
            }
        }
    });
}

function updateChartRange(range) {
    const data = window.FULL_DATA_CACHE[currentTicker];
    if (!data) return;

    // Check if we should switch to Intraday
    if (['1d', '3d', '7d', '1m'].includes(range) && data.intraday) {
        renderIntradayChart(data, range);
        return;
    }

    // Else render Daily and Zoom
    // If chart is currently Intraday (check title or dataset label?), re-render Daily
    const isCurrentlyIntraday = detailedChart && detailedChart.data.datasets[0].label === 'Intraday Price';

    if (isCurrentlyIntraday || !detailedChart) {
        renderDetailedChart(data);
    }

    // ... Existing updateChartRange logic for Daily Zoom ...
    applyDailyZoom(range);
}

function applyDailyZoom(range) {
    if (!detailedChart) return;

    const labels = detailedChart.data.labels;
    // Dataset 0 is Price (History). Find last non-null index.
    const historyData = detailedChart.data.datasets[0].data;
    let lastHistoryIdx = 0;
    for (let i = historyData.length - 1; i >= 0; i--) {
        if (historyData[i] !== null) {
            lastHistoryIdx = i;
            break;
        }
    }

    let lookbackDays = 0;
    let futureDays = 10;

    switch (range) {
        case '1d': lookbackDays = 1; futureDays = 1; break;
        case '3d': lookbackDays = 3; futureDays = 3; break;
        case '7d': lookbackDays = 7; futureDays = 7; break;
        case '1m': lookbackDays = 22; futureDays = 10; break;
        case 'All': lookbackDays = lastHistoryIdx; futureDays = 10; break;
    }

    let startIdx = lastHistoryIdx - lookbackDays;
    if (startIdx < 0) startIdx = 0;

    let endIdx = lastHistoryIdx + futureDays;
    if (endIdx >= labels.length) endIdx = labels.length - 1;

    const minLabel = labels[startIdx];
    const maxLabel = labels[endIdx];

    detailedChart.options.scales.x.min = minLabel;
    detailedChart.options.scales.x.max = maxLabel;

    detailedChart.options.plugins.title.text = 'Long Term Trend & Key Inflection Points (Daily)';

    if (range === '1d' || range === '3d' || range === '7d') {
        detailedChart.data.datasets.forEach(ds => {
            if (ds.type === 'line') ds.pointRadius = 3;
        });
    } else {
        detailedChart.data.datasets.forEach(ds => {
            if (ds.type === 'line') ds.pointRadius = 0;
        });
    }

    detailedChart.update();
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
    // Update Labels
    // Use data.dates if available, otherwise indices
    let labels;
    if (data.dates) {
        labels = data.dates; // Strings "YYYY-MM-DD"
    } else {
        // Fallback
        labels = Array.from({ length: prices.length }, (_, i) => i);
    }

    // Technicals
    const ma15 = data.technicals.mas.ma15;
    const ma30 = data.technicals.mas.ma30;
    const ma60 = data.technicals.mas.ma60;
    const ma180 = data.technicals.mas.ma180;

    // Trend Points - indices match labels array
    const trendPoints = data.technicals.trend_points;
    const scatterData = trendPoints.map(p => ({
        x: data.dates ? data.dates[p.index] : p.index, // Use date string for X if labels are dates
        y: p.value,
        type: p.type
    }));

    // Wait, if labels are strings, x in scatter should be the string or index?
    // Chart.js Category axis uses index or string matching. 
    // It's safer to use index for X if we just rely on the labels array order.
    // Actually, simply using p.index works perfectly fine with category axis.
    // Chart.js maps index 0 to label[0].
    // BUT if we ZOOM by slicing labels or setting min/max, we need to be careful.
    // If we use 'category' axis, min/max are indices? Or labels?
    // With strings, min/max are labels.

    // Let's keep it simple: Use indices for the scatter points internally to match data arrays
    // But wait, the dataset for line uses 'data: prices'. This implicitly maps to labels[0..N].
    // If we set min/max on x-axis, it filters the view.

    // So scatter data 'x' should correspond to the value on x-axis.
    // If x-axis is category (dates), x should be the date string.

    const scatterDataCorrected = trendPoints.map(p => ({
        x: data.dates ? data.dates[p.index] : p.index,
        y: p.value,
        type: p.type
    }));

    // Separate Peaks and Valleys for styling
    const peaks = scatterDataCorrected.filter(d => d.type === 'peak' || d.type === 'peark');
    const valleys = scatterDataCorrected.filter(d => d.type === 'valley');

    // Forecast Data Construction
    // History: indices 0 to N-1
    // Forecast: indices N to N+M-1 (since labels include forecast dates)
    const historyLen = prices.length;
    const forecastLen = data.forecast.length; // Should be 10
    const totalLen = labels.length; // Should match historyLen + forecastLen

    // Construct Forecast Dataset: [null...null, lastHistoryValue, pred1, pred2...]
    // Note: Chart.js needs nulls to align data to the correct labels
    const forecastData = new Array(historyLen - 1).fill(null);
    forecastData.push(prices[historyLen - 1]); // Connect to last history point
    forecastData.push(...data.forecast);

    // Construct History Dataset: [p1, p2... pN, null...null]
    const historyData = prices.concat(new Array(forecastLen).fill(null));

    detailedChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Price',
                    data: historyData,
                    borderColor: isLight ? '#0f172a' : '#f8fafc',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1,
                    order: 1
                },
                {
                    label: 'Forecast',
                    data: forecastData,
                    borderColor: '#4ade80', /* Green */
                    borderWidth: 2,
                    pointRadius: 2,
                    tension: 0.4,
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
                    backgroundColor: '#10b981',
                    pointRadius: 6,
                    pointStyle: 'triangle',
                    order: 0
                },
                {
                    label: 'Valleys',
                    data: valleys,
                    type: 'scatter',
                    backgroundColor: '#ef4444',
                    pointRadius: 6,
                    pointStyle: 'triangle',
                    rotation: 180,
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
                    display: true,
                    grid: { display: false },
                    ticks: {
                        color: textColor,
                        maxTicksLimit: 10
                    }
                },
                y: {
                    grid: { color: gridColor },
                    ticks: { color: textColor }
                }
            }
        }
    });
}

function updateChartRange(range) {
    if (!detailedChart) return;

    const labels = detailedChart.data.labels;
    // Dataset 0 is Price (History). Find last non-null index.
    const historyData = detailedChart.data.datasets[0].data;
    let lastHistoryIdx = 0;
    for (let i = historyData.length - 1; i >= 0; i--) {
        if (historyData[i] !== null) {
            lastHistoryIdx = i;
            break;
        }
    }

    let lookbackDays = 0;
    let futureDays = 10; // Default max forecast

    // Define lookback and future limit based on range
    switch (range) {
        case '1d':
            lookbackDays = 1;
            futureDays = 1; // Balance view
            break;
        case '3d':
            lookbackDays = 3;
            futureDays = 3;
            break;
        case '7d':
            lookbackDays = 7;
            futureDays = 7;
            break;
        case '1m':
            lookbackDays = 22; // ~1 month trading
            futureDays = 10; // Show all available
            break;
        case 'All':
            lookbackDays = lastHistoryIdx; // Start from 0
            futureDays = 10;
            break;
    }

    // Calculate Indices
    let startIdx = lastHistoryIdx - lookbackDays;
    if (startIdx < 0) startIdx = 0;

    // Calculate End Index (Last History + Future Limit)
    // Note: Forecast data starts at lastHistoryIdx. 
    // If we want to show 'futureDays' amount of forecast...
    // The forecast array has 10 points extending from lastHistoryIdx.
    // So target end index = lastHistoryIdx + futureDays.
    let endIdx = lastHistoryIdx + futureDays;
    if (endIdx >= labels.length) endIdx = labels.length - 1;

    // Apply to Scale
    const minLabel = labels[startIdx];
    const maxLabel = labels[endIdx];

    detailedChart.options.scales.x.min = minLabel;
    detailedChart.options.scales.x.max = maxLabel;

    // Point Radius Logic
    if (range === '1d' || range === '3d' || range === '7d') {
        detailedChart.data.datasets.forEach(ds => {
            if (ds.type === 'line') ds.pointRadius = 3;
        });
    } else {
        detailedChart.data.datasets.forEach(ds => {
            if (ds.type === 'line') ds.pointRadius = 0;
        });
    }

    detailedChart.update();
}
