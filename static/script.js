// Global state
let modelInfo = {};
let predictionChart = null;

// Material colors for consistency
const MATERIAL_COLORS = {
    'Concrete': '#34495e',
    'Glass': '#3498db',
    'Steel': '#e74c3c',
    'Wood': '#8b6f47',
    'Brick': '#c0392b'
};

// Initialize the application
document.addEventListener('DOMContentLoaded', async function() {
    await loadModelInfo();
    setupEventListeners();
});

// Load model information from API
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        if (!response.ok) throw new Error('Failed to load model info');
        
        modelInfo = await response.json();
        populateSelectOptions();
        displayModelInfo();
    } catch (error) {
        console.error('Error loading model info:', error);
        showError('Failed to load model information. Please refresh the page.');
    }
}

// Populate select options from model info
function populateSelectOptions() {
    if (!modelInfo.options) return;

    // Typology
    const typologySelect = document.getElementById('typology');
    modelInfo.options.Typology?.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option;
        opt.textContent = formatOptionLabel(option);
        typologySelect.appendChild(opt);
    });

    // Primary Code
    const primaryCodeSelect = document.getElementById('primaryCode');
    modelInfo.options['Primary Code']?.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option;
        opt.textContent = formatOptionLabel(option);
        primaryCodeSelect.appendChild(opt);
    });

    // Location Code
    // Country
    const locationSelect = document.getElementById('locationCode');
    modelInfo.options.Country?.forEach(option => {
        const opt = document.createElement('option');
        opt.value = option;
        opt.textContent = formatOptionLabel(option);
        locationSelect.appendChild(opt);
    });

    // Set first non-empty option as selected
    if (modelInfo.options.Typology?.length > 0) {
        typologySelect.value = modelInfo.options.Typology[0];
    }
    if (modelInfo.options['Primary Code']?.length > 0) {
        primaryCodeSelect.value = modelInfo.options['Primary Code'][0];
    }
        if (modelInfo.options.Country?.length > 0) {
        locationSelect.value = modelInfo.options.Country[0];
    }
}

// Format option labels
function formatOptionLabel(option) {
    const str = String(option);
    
    // Map known codes to readable labels
    const labelMaps = {
        'Typology': {
            'R-SFH': 'Single-Family House',
            'R-MFH': 'Multi-Family House',
            'R-AB': 'Apartment Block',
            'R-UNK': 'Residential (Unknown)',
            'NR-OH': 'Office (High)',
            'NR-OL': 'Office (Low)',
            'NR-C': 'Commercial (Retail/Mall)',
            'NR-E': 'Education',
            'NR-I': 'Industry',
            'NR-P': 'Public/Civic',
            'NR-H': 'Hotel/Hospital',
            'NR-UNK': 'Non-residential (Unknown)',
        },
        'Primary Code': {
            'B': 'Brick',
            'BC': 'Brick-Concrete',
            'BW': 'Brick-Wood',
            'W': 'Wood',
            'C': 'Concrete',
            'CW': 'Concrete-Wood',
            'S': 'Steel',
            'SC': 'Steel-Concrete',
            'T': 'Traditional material',
        }
    };

    // Try to find a readable label
    for (const [column, map] of Object.entries(labelMaps)) {
        if (map[str]) return map[str];
    }

    return str;
}

// Display model information
function displayModelInfo() {
    const modelDescDiv = document.getElementById('modelDescription');
    if (modelInfo.model_info) {
        modelDescDiv.textContent = modelInfo.model_info;
    }
    if (modelInfo.best_params && Object.keys(modelInfo.best_params).length > 0) {
        const params = modelInfo.best_params;
        const paramsText = Object.entries(params)
            .map(([k, v]) => `${k}: ${v}`)
            .join(', ');
        modelDescDiv.textContent += ` | Best params: ${paramsText}`;
    }
}

// Setup event listeners
function setupEventListeners() {
    const form = document.getElementById('predictionForm');
    form.addEventListener('submit', handleFormSubmit);

    const numSamplesInput = document.getElementById('numSamples');
    numSamplesInput.addEventListener('input', function() {
        document.getElementById('numSamplesValue').textContent = this.value;
    });
}

// Handle form submission
async function handleFormSubmit(event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    const data = Object.fromEntries(formData);

    // Validation
    if (!data.construction_period || !data.typology || !data.primary_code || 
        !data.hybrid_structure || !data.country_norm) {
        showError('Please fill in all fields');
        return;
    }

    // Show loading spinner, hide results
    document.getElementById('loadingSpinner').classList.remove('hidden');
    document.getElementById('resultsContainer').classList.add('hidden');
    document.getElementById('errorContainer').classList.add('hidden');

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }

        const result = await response.json();
        displayResults(result);
        document.getElementById('loadingSpinner').classList.add('hidden');
        document.getElementById('resultsContainer').classList.remove('hidden');

    } catch (error) {
        console.error('Prediction error:', error);
        document.getElementById('loadingSpinner').classList.add('hidden');
        showError(error.message);
    }
}

// Display prediction results
function displayResults(result) {
    // Display input summary
    const inputSummary = document.getElementById('inputSummary');
    inputSummary.innerHTML = '';
    
    const inputLabels = {
        'construction_period': 'Construction Period',
        'typology': 'Building Function',
        'primary_code': 'Structural System',
        'hybrid_structure': 'Hybrid Structure',
        'country_norm': 'Country'
    };

    for (const [key, label] of Object.entries(inputLabels)) {
        const value = result.input[key];
        const displayValue = formatOptionLabel(value);
        const summaryItem = document.createElement('div');
        summaryItem.className = 'summary-item';
        summaryItem.innerHTML = `<strong>${label}</strong><span>${displayValue}</span>`;
        inputSummary.appendChild(summaryItem);
    }

    // Display material predictions
    const materialsMetrics = document.getElementById('materialsMetrics');
    materialsMetrics.innerHTML = '';

    const predictions = result.predictions;
    for (const [material, values] of Object.entries(predictions)) {
        const card = document.createElement('div');
        card.className = 'material-card';
        card.innerHTML = `
            <h4>${material}</h4>
            <div class="metric">
                <div class="metric-label">5th percentile</div>
                <div class="metric-value">${values.p5.toFixed(2)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Median</div>
                <div class="metric-value">${values.p50.toFixed(2)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">95th percentile</div>
                <div class="metric-value">${values.p95.toFixed(2)}</div>
            </div>
        `;
        materialsMetrics.appendChild(card);
    }

    // Create chart
    createPredictionChart(predictions);
}

// Create prediction chart using Chart.js
function createPredictionChart(predictions) {
    const ctx = document.getElementById('predictionChart').getContext('2d');

    const materials = Object.keys(predictions);
    const medians = materials.map(m => predictions[m].p50);
    const lowerBounds = materials.map(m => predictions[m].p5);
    const upperBounds = materials.map(m => predictions[m].p95);

    const colors = materials.map(m => MATERIAL_COLORS[m] || '#3498db');

    // Destroy existing chart if it exists
    if (predictionChart) {
        predictionChart.destroy();
    }

    predictionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: materials,
            datasets: [
                {
                    label: 'Median',
                    data: medians,
                    backgroundColor: colors,
                    borderColor: colors,
                    borderWidth: 2,
                    borderRadius: 4,
                    categoryPercentage: 0.6,
                    barPercentage: 0.8,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    callbacks: {
                        afterLabel: function(context) {
                            const materialIndex = context.dataIndex;
                            const material = materials[materialIndex];
                            return [
                                `p5: ${lowerBounds[materialIndex].toFixed(2)}`,
                                `p95: ${upperBounds[materialIndex].toFixed(2)}`
                            ];
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Material Intensity (kg/m²)'
                    }
                }
            }
        },
        plugins: [{
            id: 'customErrorBars',
            afterDatasetsDraw(chart) {
                const ctx = chart.ctx;
                chart.data.datasets.forEach((dataset, i) => {
                    const meta = chart.getDatasetMeta(i);
                    meta.data.forEach((datapoint, index) => {
                        const {x, y} = datapoint.getProps(['x', 'y']);
                        const lower = lowerBounds[index];
                        const upper = upperBounds[index];
                        const median = medians[index];

                        // Convert data values to pixel values
                        const yScalePixel = chart.scales.y;
                        const lowerPixel = yScalePixel.getPixelForValue(lower);
                        const upperPixel = yScalePixel.getPixelForValue(upper);

                        // Draw error bar
                        ctx.strokeStyle = '#666';
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(x, upperPixel);
                        ctx.lineTo(x, lowerPixel);
                        ctx.stroke();

                        // Draw caps
                        const capWidth = 8;
                        ctx.beginPath();
                        ctx.moveTo(x - capWidth, upperPixel);
                        ctx.lineTo(x + capWidth, upperPixel);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.moveTo(x - capWidth, lowerPixel);
                        ctx.lineTo(x + capWidth, lowerPixel);
                        ctx.stroke();
                    });
                });
            }
        }]
    });

    // Ensure container has a height
    const container = document.querySelector('.chart-container');
    container.style.position = 'relative';
    container.style.height = '350px';
}

// Show error message
function showError(message) {
    const errorContainer = document.getElementById('errorContainer');
    const errorMessage = document.getElementById('errorMessage');
    errorMessage.textContent = message;
    document.getElementById('loadingSpinner').classList.add('hidden');
    document.getElementById('resultsContainer').classList.add('hidden');
    errorContainer.classList.remove('hidden');
}
