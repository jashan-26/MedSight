/* ==========================================================================
   PACS AI RADIOLOGY WEB WORKSTATION - JAVASCRIPT FRONTEND CONTROLLER
   ========================================================================== */

let selectedFile = null;
let chartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    fetchSamples();
    setupDragAndDrop();
});

async function fetchSamples() {
    try {
        const response = await fetch("/api/samples");
        const data = await response.json();
        const select = document.getElementById("sampleSelect");
        
        data.samples.forEach(sample => {
            const opt = document.createElement("option");
            opt.value = sample;
            opt.textContent = sample;
            select.appendChild(opt);
        });
    } catch (err) {
        console.error("Error fetching samples:", err);
    }
}

function setupDragAndDrop() {
    const dropzone = document.getElementById("dropzone");

    ["dragenter", "dragover"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = "#FDFBF7";
            dropzone.style.background = "#1A284C";
        });
    });

    ["dragleave", "drop"].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.style.borderColor = "#EADBC8";
            dropzone.style.background = "#070D1E";
        });
    });

    dropzone.addEventListener("drop", (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            selectedFile = files[0];
            document.getElementById("fileNameDisplay").innerText = selectedFile.name;
            document.getElementById("sampleSelect").value = "None";
        }
    });
}

function handleFileSelect(event) {
    if (event.target.files.length > 0) {
        selectedFile = event.target.files[0];
        document.getElementById("fileNameDisplay").innerText = selectedFile.name;
        document.getElementById("sampleSelect").value = "None";
    }
}

function handleSampleSelect(event) {
    if (event.target.value !== "None") {
        selectedFile = null;
        document.getElementById("fileNameDisplay").innerText = "PNG, JPG, or JPEG formats";
        document.getElementById("fileInput").value = "";
    }
}

async function runInference() {
    const sampleName = document.getElementById("sampleSelect").value;

    if (!selectedFile && sampleName === "None") {
        alert("Please upload a Chest X-ray file or select a pre-loaded sample.");
        return;
    }

    // UI Loading State
    document.getElementById("placeholderView").style.display = "none";
    document.getElementById("resultsView").style.display = "none";
    document.getElementById("loadingSpinner").style.display = "block";

    const formData = new FormData();
    if (selectedFile) {
        formData.append("file", selectedFile);
    }
    formData.append("sample_name", sampleName);
    formData.append("model_name", document.getElementById("modelSelect").value);
    formData.append("auto_crop", document.getElementById("autoCropCheck").checked);
    formData.append("enable_clahe", document.getElementById("claheCheck").checked);
    formData.append("colormap", document.getElementById("colormapSelect").value);
    formData.append("alpha", document.getElementById("alphaSlider").value);

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Server returned error status ${response.status}`);
        }

        const data = await response.json();
        renderResults(data);

    } catch (err) {
        alert("Inference Error: " + err.message);
    } finally {
        document.getElementById("loadingSpinner").style.display = "none";
    }
}

function renderResults(data) {
    document.getElementById("resultsView").style.display = "block";

    // KPIs
    document.getElementById("resPrimary").innerText = data.primary_finding;
    document.getElementById("resConfidence").innerText = `${data.confidence_pct}%`;
    document.getElementById("resRating").innerText = `Rating: ${data.uncertainty.confidence_rating}`;
    document.getElementById("resModel").innerText = data.model_name;
    document.getElementById("resAnatomy").innerText = data.anatomical_attention;

    // Risk Badge
    const riskBadgeContainer = document.getElementById("resRiskBadge");
    const risk = data.risk_level;
    riskBadgeContainer.innerHTML = `<span class="risk-badge risk-${risk}">${risk} RISK</span>`;

    // Images
    document.getElementById("imgPreprocessed").src = data.images.preprocessed;
    document.getElementById("imgHeatmap").src = data.images.heatmap;
    document.getElementById("imgOverlay").src = data.images.overlay;

    // Chart.js Bar Chart in Navy Blue & Cream Color Ramps
    renderProbabilityChart(data.probabilities);
}

function renderProbabilityChart(probDict) {
    const ctx = document.getElementById("probChart").getContext("2d");

    if (chartInstance) {
        chartInstance.destroy();
    }

    const labels = Object.keys(probDict);
    const values = Object.values(probDict).map(v => (v * 100).toFixed(1));

    // Gradient fill (Navy Blue to Gold Cream)
    const gradient = ctx.createLinearGradient(0, 0, 400, 0);
    gradient.addColorStop(0, '#1E3A8A');
    gradient.addColorStop(0.5, '#2563EB');
    gradient.addColorStop(1, '#EADBC8');

    chartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Probability (%)',
                data: values,
                backgroundColor: gradient,
                borderColor: '#EADBC8',
                borderWidth: 1,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    min: 0,
                    max: 100,
                    ticks: { color: '#FAF6EE' },
                    grid: { color: 'rgba(234, 219, 200, 0.15)' }
                },
                y: {
                    ticks: { color: '#FAF6EE', font: { family: 'Inter', weight: '600' } },
                    grid: { color: 'rgba(234, 219, 200, 0.15)' }
                }
            }
        }
    });
}

async function downloadReport() {
    const sampleName = document.getElementById("sampleSelect").value;
    const modelName = document.getElementById("modelSelect").value;

    const formData = new FormData();
    if (selectedFile) {
        formData.append("file", selectedFile);
    }
    formData.append("sample_name", sampleName);
    formData.append("model_name", modelName);

    try {
        const response = await fetch("/api/report", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Failed to generate report PDF.");
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `PACS_Report_${modelName}.pdf`;
        document.body.appendChild(a);
        a.click();
        a.remove();
    } catch (err) {
        alert("Error downloading PDF: " + err.message);
    }
}
