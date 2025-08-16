// Update range input values
const brewTimeEl = document.getElementById('brew_time');
const waterTempEl = document.getElementById('water_temp');
const ratioEl = document.getElementById('coffee_water_ratio');
const acidityEl = document.getElementById('acidity_pref');
const bitternessEl = document.getElementById('bitterness_pref');

if (brewTimeEl) brewTimeEl.addEventListener('input', function() {
	document.getElementById('brew_time_value').textContent = this.value + 's';
});

if (waterTempEl) waterTempEl.addEventListener('input', function() {
	document.getElementById('water_temp_value').textContent = this.value + '°C';
});

if (ratioEl) ratioEl.addEventListener('input', function() {
	document.getElementById('ratio_value').textContent = '1:' + this.value;
});

if (acidityEl) acidityEl.addEventListener('input', function() {
	document.getElementById('acidity_pref_value').textContent = this.value;
});

if (bitternessEl) bitternessEl.addEventListener('input', function() {
	document.getElementById('bitterness_pref_value').textContent = this.value;
});

// Handle form submission
const formEl = document.getElementById('coffeeForm');
if (formEl) formEl.addEventListener('submit', async function(e) {
	e.preventDefault();
	console.log('Form submitted!'); // Debug logging
	
	// Show loading state
	const submitBtn = document.getElementById('predictBtn');
	const originalBtnText = submitBtn.innerHTML;
	submitBtn.disabled = true;
	submitBtn.innerHTML = '<i class="fas fa-mug-hot"></i> Predicting...';
	document.getElementById('loading').style.display = 'block';
	
	try {
		// Get form data mapped to backend expectations
		const formData = new URLSearchParams({
			brewing_method: document.getElementById('brewing_method').value,
			bean_type: document.getElementById('bean_type').value,
			roast_level: document.getElementById('roast_level').value,
			grind_size: document.getElementById('grind_size').value,
			water_temp: document.getElementById('water_temp').value,
			brew_time: document.getElementById('brew_time').value,
			coffee_water_ratio: (1 / parseFloat(document.getElementById('coffee_water_ratio').value)).toFixed(4),
			acidity_pref: document.getElementById('acidity_pref').value,
			bitterness_pref: document.getElementById('bitterness_pref').value
		});
		
		console.log('Form data:', formData.toString()); // Debug logging
		
		// Send prediction request
		const response = await fetch('/predict', {
			method: 'POST',
			headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
			body: formData
		});
		
		const data = await response.json();
		
		console.log('Prediction response:', data); // Debug logging
		
		const resultsDiv = document.getElementById('results');
		console.log('Results div found:', resultsDiv); // Debug logging
		
		if (data.success) {
			console.log('Prediction successful, displaying results...'); // Debug logging
			// Display results
			document.getElementById('predictionScore').textContent = data.prediction.score;
			document.getElementById('confidence').textContent = data.prediction.confidence;
			document.getElementById('interpretation').textContent = data.prediction.interpretation || '';
			
			// Build flavor chips from interpretation keywords
			const chips = buildFlavorChips(data.prediction.interpretation);
			const chipsContainer = document.getElementById('flavorChips');
			chipsContainer.innerHTML = '';
			chips.forEach(ch => chipsContainer.appendChild(ch));
			
			// Force show the results div with inline styles
			resultsDiv.style.setProperty('display', 'block', 'important');
			resultsDiv.style.visibility = 'visible';
			resultsDiv.style.opacity = '1';
			resultsDiv.scrollIntoView({ behavior: 'smooth' });
			console.log('Results displayed successfully'); // Debug logging
		} else {
			console.log('Prediction failed:', data.error); // Debug logging
			resultsDiv.innerHTML = `
				<div class="card">
					<h3>Something went wrong</h3>
					<p>${data.error || 'An unknown error occurred'}</p>
				</div>
			`;
			resultsDiv.style.display = 'block';
			resultsDiv.scrollIntoView({ behavior: 'smooth' });
			console.error('Prediction error:', data.error);
		}
	} catch (error) {
		alert('Error: ' + error.message);
		console.error('Prediction error:', error);
	} finally {
		// Reset button state
		submitBtn.disabled = false;
		submitBtn.innerHTML = originalBtnText;
		document.getElementById('loading').style.display = 'none';
	}
});

function buildFlavorChips(interpretation) {
	const keywords = [
		{ key: 'chocolate', icon: 'fa-solid fa-square', color: '#6F4E37' },
		{ key: 'cocoa', icon: 'fa-solid fa-square', color: '#5a3a2c' },
		{ key: 'caramel', icon: 'fa-solid fa-square', color: '#B87333' },
		{ key: 'nut', icon: 'fa-solid fa-square', color: '#a67c52' },
		{ key: 'hazelnut', icon: 'fa-solid fa-square', color: '#a67c52' },
		{ key: 'almond', icon: 'fa-solid fa-square', color: '#a67c52' },
		{ key: 'citrus', icon: 'fa-solid fa-lemon', color: '#d4a514' },
		{ key: 'berry', icon: 'fa-solid fa-seedling', color: '#9CAF88' },
		{ key: 'floral', icon: 'fa-solid fa-seedling', color: '#9CAF88' },
		{ key: 'spice', icon: 'fa-solid fa-pepper-hot', color: '#b65c42' }
	];
	const chips = [];
	const text = (interpretation || '').toLowerCase();
	keywords.forEach(k => {
		if (text.includes(k.key)) {
			chips.push(createChip(k.icon, capitalize(k.key), k.color));
		}
	});
	if (chips.length === 0) {
		chips.push(createChip('fa-solid fa-mug-hot', 'Balanced', '#9CAF88'));
	}
	return chips;
}

function createChip(iconClass, label, color) {
	const chip = document.createElement('span');
	chip.className = 'flavor-chip';
	chip.innerHTML = `<i class="${iconClass}" style="color:${color}"></i> ${label}`;
	return chip;
}

function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
