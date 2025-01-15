document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayResults(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error making prediction: ' + error.message);
    }
});

function displayResults(result) {
    // Create results container if it doesn't exist
    let resultsDiv = document.getElementById('results');
    if (!resultsDiv) {
        resultsDiv = document.createElement('div');
        resultsDiv.id = 'results';
        resultsDiv.className = 'results';
        document.querySelector('main').appendChild(resultsDiv);
    }
    
    // Clear previous results
    resultsDiv.innerHTML = `
        <div class="results-header">
            <h2>Risk Assessment Results</h2>
            <button onclick="closeResults()" class="close-btn">&times;</button>
        </div>
        
        <div class="risk-indicator">
            <div class="risk-level">
                <h3>Risk Level</h3>
                <span class="risk-badge ${result.risk_level.toLowerCase()}">${result.risk_level}</span>
            </div>
            <div class="risk-probability">
                <h3>Dropout Probability</h3>
                <div class="progress-bar">
                    <div class="progress" style="width: ${result.dropout_probability * 100}%"></div>
                    <span id="probabilityValue">${(result.dropout_probability * 100).toFixed(1)}%</span>
                </div>
            </div>
        </div>
        
        <div class="risk-factors">
            <h3>Risk Factors</h3>
            <ul id="riskFactors"></ul>
        </div>
        
        <div class="recommendations">
            <h3>Intervention Plan</h3>
            <div class="intervention-timeline">
                <div class="timeline-section">
                    <h4>Immediate Actions</h4>
                    <ul id="immediateActions"></ul>
                </div>
                <div class="timeline-section">
                    <h4>Medium Term Actions</h4>
                    <ul id="mediumTermActions"></ul>
                </div>
                <div class="timeline-section">
                    <h4>Long Term Actions</h4>
                    <ul id="longTermActions"></ul>
                </div>
            </div>
        </div>
    `;
    
    // Populate risk factors
    const riskFactorsList = document.getElementById('riskFactors');
    result.risk_factors.forEach(factor => {
        const li = document.createElement('li');
        li.innerHTML = `
            <strong>${factor.category}</strong> - ${factor.risk}
            <br>
            <small>Severity: ${factor.severity}</small>
        `;
        riskFactorsList.appendChild(li);
    });
    
    // Populate intervention plans
    document.getElementById('immediateActions').innerHTML = 
        result.intervention_plan.immediate_actions.map(action => `<li>${action}</li>`).join('');
    document.getElementById('mediumTermActions').innerHTML = 
        result.intervention_plan.medium_term_actions.map(action => `<li>${action}</li>`).join('');
    document.getElementById('longTermActions').innerHTML = 
        result.intervention_plan.long_term_actions.map(action => `<li>${action}</li>`).join('');
    
    resultsDiv.classList.remove('hidden');
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

function closeResults() {
    const resultsDiv = document.getElementById('results');
    if (resultsDiv) {
        resultsDiv.classList.add('hidden');
    }
}