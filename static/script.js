// Global variables
let detectionCount = 0;
let uniqueMudras = new Set();
let sessionStartTime = Date.now();
let mudraHistory = [];
let currentMudra = "No Mudra Detected";
let previousMudra = "No Mudra Detected";
let allMudras = [];
let isUpdating = false;  // Prevent concurrent updates
let stableFrames = 0;
let requiredStableFrames = 3;  // Mudra must be stable for 3 frames to count
let lastDetectedMudra = null;

// Update current mudra display - optimized with stability check
function updateMudraDisplay() {
    // Skip if already updating
    if (isUpdating) return;
    
    isUpdating = true;
    
    fetch('/current_mudra')
        .then(response => response.json())
        .then(data => {
            const mudraName = data.mudra;
            const mudraNameElement = document.getElementById('mudra-name');
            const statusText = document.getElementById('status-text');
            
            // Always update display immediately for responsiveness
            mudraNameElement.textContent = mudraName;
            
            // Check if mudra changed
            if (mudraName !== currentMudra) {
                // Reset stability counter when mudra changes
                stableFrames = 0;
                previousMudra = currentMudra;
                currentMudra = mudraName;
            } else {
                // Same mudra detected, increment stability counter
                stableFrames++;
            }
            
            // Update styling based on current mudra
            if (mudraName === "No Mudra Detected") {
                mudraNameElement.className = 'mudra-name no-detection';
                statusText.textContent = 'Waiting...';
            } else {
                mudraNameElement.className = 'mudra-name detected';
                
                // Update stability indicator
                updateStabilityIndicator();
                
                // Only count as a NEW detection if:
                // 1. Mudra is stable (seen for required frames)
                // 2. Different from the last counted detection
                // 3. Previous state was "No Mudra Detected" OR different mudra
                if (stableFrames === requiredStableFrames && mudraName !== lastDetectedMudra) {
                    // This is a valid new detection
                    detectionCount++;
                    uniqueMudras.add(mudraName);
                    lastDetectedMudra = mudraName;
                    updateStats();
                    
                    // Add to history
                    addToHistory(mudraName);
                    
                    // Highlight in mudra grid
                    highlightMudra(mudraName);
                }
            }
            
            // Reset last detected mudra when hand is removed
            if (mudraName === "No Mudra Detected" && stableFrames > requiredStableFrames) {
                lastDetectedMudra = null;
            }
        })
        .catch(error => {
            console.error('Error fetching mudra:', error);
        })
        .finally(() => {
            isUpdating = false;
        });
}

// Update session statistics
function updateStats() {
    document.getElementById('detection-count').textContent = detectionCount;
    document.getElementById('unique-mudras').textContent = uniqueMudras.size;
}

// Update stability indicator
function updateStabilityIndicator() {
    const statusText = document.getElementById('status-text');
    
    if (currentMudra === "No Mudra Detected") {
        statusText.textContent = 'Waiting...';
        return;
    }
    
    if (stableFrames < requiredStableFrames) {
        // Still stabilizing
        statusText.textContent = `Stabilizing... (${stableFrames}/${requiredStableFrames})`;
    } else {
        // Stable detection
        statusText.textContent = 'Stable Detection ✓';
    }
}

// Update session time
function updateSessionTime() {
    const elapsed = Math.floor((Date.now() - sessionStartTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    const timeString = `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
    document.getElementById('session-time').textContent = timeString;
}

// Add mudra to history
function addToHistory(mudraName) {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    
    // Add to history array (keep last 10)
    mudraHistory.unshift({
        name: mudraName,
        time: timeString
    });
    
    if (mudraHistory.length > 10) {
        mudraHistory.pop();
    }
    
    // Update history display
    const historyList = document.getElementById('history-list');
    historyList.innerHTML = '';
    
    mudraHistory.forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        historyItem.innerHTML = `
            <span class="history-mudra">${item.name}</span>
            <span class="history-time">${item.time}</span>
        `;
        historyList.appendChild(historyItem);
    });
}

// Highlight detected mudra in grid
function highlightMudra(mudraName) {
    // Remove previous highlights
    document.querySelectorAll('.mudra-card').forEach(card => {
        card.classList.remove('detected');
    });
    
    // Add highlight to current mudra
    const mudraCards = document.querySelectorAll('.mudra-card');
    mudraCards.forEach(card => {
        if (card.textContent === mudraName) {
            card.classList.add('detected');
            setTimeout(() => {
                card.classList.remove('detected');
            }, 2000);
        }
    });
}

// Load and display all supported mudras
function loadMudraList() {
    fetch('/mudra_list')
        .then(response => response.json())
        .then(data => {
            allMudras = data.mudras;
            const mudrasGrid = document.getElementById('mudras-grid');
            mudrasGrid.innerHTML = '';
            
            data.mudras.forEach(mudra => {
                const mudraCard = document.createElement('div');
                mudraCard.className = 'mudra-card';
                mudraCard.innerHTML = `
                    <div class="mudra-card-name">${mudra}</div>
                `;
                mudrasGrid.appendChild(mudraCard);
            });
        })
        .catch(error => {
            console.error('Error loading mudra list:', error);
            document.getElementById('mudras-grid').innerHTML = 
                '<p class="empty-state">Failed to load mudras</p>';
        });
}

// Toggle modals
function toggleInfo() {
    const modal = document.getElementById('info-modal');
    modal.style.display = modal.style.display === 'block' ? 'none' : 'block';
}

function toggleHelp() {
    const modal = document.getElementById('help-modal');
    modal.style.display = modal.style.display === 'block' ? 'none' : 'block';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const infoModal = document.getElementById('info-modal');
    const helpModal = document.getElementById('help-modal');
    
    if (event.target === infoModal) {
        infoModal.style.display = 'none';
    }
    if (event.target === helpModal) {
        helpModal.style.display = 'none';
    }
}

// Check video feed status
function checkVideoFeedStatus() {
    const videoFeed = document.querySelector('.video-feed');
    
    videoFeed.addEventListener('error', () => {
        console.error('Video feed error - camera may not be available');
        alert('Camera feed error! Please check if the camera is accessible.');
    });
}

// Reset statistics
function resetStats() {
    if (confirm('Are you sure you want to reset all statistics?')) {
        // Reset counters
        detectionCount = 0;
        uniqueMudras = new Set();
        mudraHistory = [];
        sessionStartTime = Date.now();
        lastDetectedMudra = null;
        stableFrames = 0;
        
        // Update display
        updateStats();
        updateSessionTime();
        
        // Clear history display
        const historyList = document.getElementById('history-list');
        historyList.innerHTML = '<p class="empty-state">No mudras detected yet</p>';
        
        // Remove highlights from mudra grid
        document.querySelectorAll('.mudra-card').forEach(card => {
            card.classList.remove('detected');
        });
        
        console.log('✅ Statistics reset successfully');
    }
}

// Initialize app
document.addEventListener('DOMContentLoaded', function() {
    console.log('🙏 Mudra Detection System Initialized');
    
    // Load mudra list
    loadMudraList();
    
    // Check video feed
    checkVideoFeedStatus();
    
    // Update mudra status every 200ms (5 FPS) - Reduced for better performance
    setInterval(updateMudraDisplay, 200);
    
    // Update session time every second
    setInterval(updateSessionTime, 1000);
    
    // Initial stats update
    updateStats();
    updateSessionTime();
});

// Handle page visibility change (pause updates when tab is not visible)
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('Page hidden - pausing updates');
    } else {
        console.log('Page visible - resuming updates');
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Press 'i' for info
    if (event.key === 'i' || event.key === 'I') {
        toggleInfo();
    }
    // Press 'h' for help
    if (event.key === 'h' || event.key === 'H') {
        toggleHelp();
    }
    // Press 'Escape' to close modals
    if (event.key === 'Escape') {
        document.getElementById('info-modal').style.display = 'none';
        document.getElementById('help-modal').style.display = 'none';
    }
});

// Export functions for inline HTML usage
window.toggleInfo = toggleInfo;
window.toggleHelp = toggleHelp;
window.resetStats = resetStats;
