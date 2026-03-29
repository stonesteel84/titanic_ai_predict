// Optional: Load background stars/water particles for extra Wow effect
tsParticles.loadPreset("sea", {
    background: { color: "transparent" },
    particles: { color: { value: "#64ffda" } }
}).then(container => {
    tsParticles.load("tsparticles", {
        preset: "sea",
    });
});

document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = e.target.querySelector('button');
    const originalText = btn.textContent;
    btn.textContent = 'Calculating Fate...';
    btn.style.opacity = '0.7';

    // Prepare payload for FastAPI backend
    const payload = {
        pclass: parseInt(document.getElementById('pclass').value),
        sex: parseInt(document.getElementById('sex').value),
        age: parseFloat(document.getElementById('age').value),
        sibsp: parseInt(document.getElementById('sibsp').value),
        parch: parseInt(document.getElementById('parch').value),
        embarked: document.getElementById('embarked').value
    };

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await res.json();
        
        if (data.error) {
            alert('Error: ' + data.error);
            return;
        }

        const resultDiv = document.getElementById('result');
        const resultText = document.getElementById('result-text');
        const probBar = document.getElementById('prob-bar');
        const probText = document.getElementById('prob-text');

        // Hide old results to replay animation
        resultDiv.classList.add('hidden');
        probBar.style.width = '0%';
        
        setTimeout(() => {
            resultDiv.classList.remove('hidden');
            
            if (data.survived) {
                resultText.textContent = 'Survived (생존 예상)';
                resultText.className = 'survived';
            } else {
                resultText.textContent = 'Perished (사망 예상)';
                resultText.className = 'perished';
            }

            // Animate progress bar
            setTimeout(() => {
                probBar.style.width = `${data.probability}%`;
            }, 100);

            probText.innerHTML = `Survival Probability: <strong>${data.probability}%</strong>`;
        }, 100); // Wait for CSS DOM refresh
        
    } catch (err) {
        alert('Server connection error. Please try again.');
        console.error(err);
    } finally {
        btn.textContent = originalText;
        btn.style.opacity = '1';
    }
});
