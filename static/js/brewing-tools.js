// Brewing Timer
class BrewingTimer {
    constructor() {
        this.timer = null;
        this.remainingTime = 0;
        this.isRunning = false;
        this.notificationSound = new Audio('https://assets.mixkit.co/sfx/preview/mixkit-alarm-digital-clock-beep-989.mp3');
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        // Timer elements
        this.timerDisplay = document.getElementById('timer-display');
        this.timerInput = document.getElementById('timer-input');
        this.startTimerBtn = document.getElementById('start-timer');
        this.pauseTimerBtn = document.getElementById('pause-timer');
        this.resetTimerBtn = document.getElementById('reset-timer');
        this.timerPresets = document.querySelectorAll('.timer-preset');
    }

    setupEventListeners() {
        if (this.startTimerBtn) {
            this.startTimerBtn.addEventListener('click', () => this.startTimer());
        }
        if (this.pauseTimerBtn) {
            this.pauseTimerBtn.addEventListener('click', () => this.pauseTimer());
        }
        if (this.resetTimerBtn) {
            this.resetTimerBtn.addEventListener('click', () => this.resetTimer());
        }
        
        // Add preset timer buttons
        this.timerPresets.forEach(preset => {
            preset.addEventListener('click', (e) => {
                const minutes = parseInt(e.target.dataset.minutes);
                this.setTimer(minutes * 60);
            });
        });

        // Request notification permission
        if ('Notification' in window) {
            Notification.requestPermission();
        }
    }

    setTimer(seconds) {
        this.remainingTime = seconds;
        this.updateDisplay();
    }

    startTimer() {
        if (this.isRunning) return;
        
        if (this.remainingTime <= 0) {
            const inputMinutes = parseInt(this.timerInput.value) || 3;
            this.remainingTime = inputMinutes * 60;
        }

        this.isRunning = true;
        this.timer = setInterval(() => this.tick(), 1000);
        this.updateButtonStates();
    }

    pauseTimer() {
        clearInterval(this.timer);
        this.isRunning = false;
        this.updateButtonStates();
    }

    resetTimer() {
        clearInterval(this.timer);
        this.isRunning = false;
        this.remainingTime = 0;
        this.timerInput.value = '3';
        this.updateDisplay();
        this.updateButtonStates();
    }

    tick() {
        if (this.remainingTime <= 0) {
            this.timerComplete();
            return;
        }
        this.remainingTime--;
        this.updateDisplay();
    }

    updateDisplay() {
        if (!this.timerDisplay) return;
        
        const minutes = Math.floor(this.remainingTime / 60);
        const seconds = this.remainingTime % 60;
        this.timerDisplay.textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    }

    updateButtonStates() {
        if (!this.startTimerBtn || !this.pauseTimerBtn) return;
        
        this.startTimerBtn.disabled = this.isRunning;
        this.pauseTimerBtn.disabled = !this.isRunning;
    }

    timerComplete() {
        clearInterval(this.timer);
        this.isRunning = false;
        this.remainingTime = 0;
        this.updateDisplay();
        this.updateButtonStates();
        this.notifyUser("Brew Time's Up!");
    }

    notifyUser(message) {
        // Play sound
        this.notificationSound.play().catch(e => console.log("Couldn't play sound:", e));
        
        // Show notification if permitted
        if (Notification.permission === 'granted') {
            new Notification('Coffee Timer', { body: message });
        }
        
        // Fallback alert
        alert(message);
    }
}

// Brewing Calculator
class BrewingCalculator {
    constructor() {
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        // Calculator elements
        this.calculatorForm = document.getElementById('calculator-form');
        this.coffeeAmount = document.getElementById('coffee-amount');
        this.waterAmount = document.getElementById('water-amount');
        this.ratio = document.getElementById('ratio');
        this.ratioValue = document.getElementById('ratio-value');
        this.calculateBtn = document.getElementById('calculate-ratio');
        this.resultDisplay = document.getElementById('calculator-result');
    }

    setupEventListeners() {
        if (this.calculatorForm) {
            this.calculatorForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.calculate();
            });
        }
        
        if (this.ratio) {
            this.ratio.addEventListener('input', () => {
                this.ratioValue.textContent = `1:${this.ratio.value}`;
            });
        }
    }

    calculate() {
        const ratioValue = parseFloat(this.ratio.value);
        
        if (this.coffeeAmount.value && !this.waterAmount.value) {
            // Calculate water amount
            const coffee = parseFloat(this.coffeeAmount.value);
            const water = coffee * ratioValue;
            this.waterAmount.value = water.toFixed(1);
            this.showResult(`For ${coffee}g of coffee, use ${water.toFixed(1)}g of water`);
        } 
        else if (!this.coffeeAmount.value && this.waterAmount.value) {
            // Calculate coffee amount
            const water = parseFloat(this.waterAmount.value);
            const coffee = water / ratioValue;
            this.coffeeAmount.value = coffee.toFixed(1);
            this.showResult(`For ${water}g of water, use ${coffee.toFixed(1)}g of coffee`);
        }
        else if (this.coffeeAmount.value && this.waterAmount.value) {
            // Calculate ratio
            const coffee = parseFloat(this.coffeeAmount.value);
            const water = parseFloat(this.waterAmount.value);
            const calculatedRatio = (water / coffee).toFixed(1);
            this.ratio.value = calculatedRatio;
            this.ratioValue.textContent = `1:${calculatedRatio}`;
            this.showResult(`Your coffee to water ratio is 1:${calculatedRatio}`);
        }
    }

    showResult(message) {
        if (!this.resultDisplay) return;
        
        this.resultDisplay.textContent = message;
        this.resultDisplay.style.display = 'block';
        
        // Hide after 5 seconds
        setTimeout(() => {
            this.resultDisplay.style.opacity = '0';
            setTimeout(() => {
                this.resultDisplay.style.display = 'none';
                this.resultDisplay.style.opacity = '1';
            }, 500);
        }, 5000);
    }
}

// Initialize tools when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Initialize timer if elements exist
    if (document.getElementById('timer-display')) {
        window.brewingTimer = new BrewingTimer();
    }
    
    // Initialize calculator if elements exist
    if (document.getElementById('calculator-form')) {
        window.brewingCalculator = new BrewingCalculator();
    }
});
