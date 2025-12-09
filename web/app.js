// Global state
let credentials = {
    username: '',
    password: ''
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Check if already authenticated (stored in sessionStorage)
    const storedAuth = sessionStorage.getItem('authenticated');
    if (storedAuth === 'true') {
        const storedUsername = sessionStorage.getItem('username');
        const storedPassword = sessionStorage.getItem('password');
        if (storedUsername && storedPassword) {
            credentials.username = storedUsername;
            credentials.password = storedPassword;
            showQueryInterface();
            // Automatically warmup after showing interface
            warmup();
        }
    }
});

// Handle login form submission
async function handleLogin(event) {
    event.preventDefault();
    
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
    const errorDiv = document.getElementById('login-error');
    
    // Clear previous errors
    errorDiv.textContent = '';
    errorDiv.classList.remove('show');
    
    // Store credentials
    credentials.username = username;
    credentials.password = password;
    
    // Test authentication by calling a protected endpoint
    try {
        const response = await fetch('/api/tools/list_drugs', {
            method: 'GET',
            headers: {
                'Authorization': 'Basic ' + btoa(`${username}:${password}`)
            }
        });
        
        if (response.ok) {
            // Authentication successful
            sessionStorage.setItem('authenticated', 'true');
            sessionStorage.setItem('username', username);
            sessionStorage.setItem('password', password);
            
            showQueryInterface();
            // Automatically call warmup after successful login
            warmup();
        } else {
            // Authentication failed
            errorDiv.textContent = 'Invalid username or password';
            errorDiv.classList.add('show');
        }
    } catch (error) {
        errorDiv.textContent = `Login error: ${error.message}`;
        errorDiv.classList.add('show');
    }
}

// Show query interface and hide login form
function showQueryInterface() {
    document.getElementById('login-form').style.display = 'none';
    document.getElementById('query-form').style.display = 'block';
}

// Handle logout
function logout() {
    sessionStorage.removeItem('authenticated');
    sessionStorage.removeItem('username');
    sessionStorage.removeItem('password');
    credentials.username = '';
    credentials.password = '';
    
    document.getElementById('login-form').style.display = 'block';
    document.getElementById('query-form').style.display = 'none';
    document.getElementById('result-container').style.display = 'none';
    document.getElementById('warmup-status').style.display = 'none';
    document.getElementById('queryForm').reset();
}

// Warmup function - called automatically after login
async function warmup() {
    const warmupStatus = document.getElementById('warmup-status');
    warmupStatus.style.display = 'block';
    warmupStatus.className = 'status-message info';
    warmupStatus.textContent = 'Warming up system...';
    
    try {
        const response = await fetch('/api/warmup', {
            method: 'POST',
            headers: {
                'Authorization': 'Basic ' + btoa(`${credentials.username}:${credentials.password}`),
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            warmupStatus.className = 'status-message success';
            if (data.status === 'already_warmed_up') {
                warmupStatus.textContent = 'System ready (already warmed up)';
            } else {
                warmupStatus.textContent = 'System warmed up and ready!';
            }
            // Hide warmup status after 3 seconds
            setTimeout(() => {
                warmupStatus.style.display = 'none';
            }, 3000);
        } else {
            const errorData = await response.json();
            warmupStatus.className = 'status-message error';
            warmupStatus.textContent = `Warmup failed: ${errorData.detail || 'Unknown error'}`;
        }
    } catch (error) {
        warmupStatus.className = 'status-message error';
        warmupStatus.textContent = `Warmup error: ${error.message}`;
    }
}

// Status messages that reflect the RAG pipeline steps
const queryStatusMessages = [
    'Greini fyrirspurn...',
    'Leita að viðeigandi lyfjaupplýsingum í gagnagrunni...',
    'Safna saman viðeigandi köflum úr SmPC skjölum...',
    'Smíða svar með tilvísunum...',
    'Staðfesti heimildir...'
];

/**
 * Show rotating status messages during query processing.
 * 
 * @param {HTMLElement} resultDiv - The div element to update with status messages
 * @returns {number} - Interval ID that can be used to clear the interval
 */
function showQueryStatus(resultDiv) {
    let messageIndex = 0;
    
    // Show initial message
    resultDiv.innerHTML = `<div class="loading">${queryStatusMessages[0]}</div>`;
    
    // Rotate through messages every 2 seconds
    const intervalId = setInterval(() => {
        messageIndex = (messageIndex + 1) % queryStatusMessages.length;
        resultDiv.innerHTML = `<div class="loading">${queryStatusMessages[messageIndex]}</div>`;
    }, 8000);
    
    return intervalId;
}

// Handle query form submission
async function handleQuery(event) {
    event.preventDefault();
    
    const question = document.getElementById('question').value;
    const drugId = document.getElementById('drug-id').value;
    const resultContainer = document.getElementById('result-container');
    const resultDiv = document.getElementById('result');
    
    // Show loading state with rotating status messages
    resultContainer.style.display = 'block';
    const statusInterval = showQueryStatus(resultDiv);
    
    try {
        const response = await fetch('/api/tools/ask_smpc', {
            method: 'POST',
            headers: {
                'Authorization': 'Basic ' + btoa(`${credentials.username}:${credentials.password}`),
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                drug_id: drugId || null,
                session_id: 'web_session'
            })
        });
        
        // Stop rotating status messages
        clearInterval(statusInterval);
        
        const data = await response.json();
        
        if (response.ok && data.answer) {
            // Display question at the top
            let html = '<div class="question-display">';
            html += `<em>${escapeHtml(question)}</em>`;
            html += '</div>';
            
            // Display answer with markdown rendering
            html += '<div class="answer-container">';
            html += '<h3>Answer:</h3>';
            // Render markdown to HTML using marked.js
            const renderedMarkdown = marked.parse(data.answer);
            html += `<div class="markdown-content">${renderedMarkdown}</div>`;
            html += '</div>';
            
            // Display sources if available
            if (data.sources && data.sources.length > 0) {
                html += '<div class="sources-container">';
                html += '<h3>Sources:</h3>';
                
                data.sources.forEach((source, index) => {
                    html += '<div class="source-item">';
                    html += `<strong>${escapeHtml(source.drug_id || 'Unknown')}</strong>`;
                    html += '<div class="source-meta">';
                    html += `Section ${escapeHtml(source.section_number || 'Unknown')}: ${escapeHtml(source.section_title || 'Unknown')}`;
                    if (source.page && source.page !== 'Unknown') {
                        html += ` | Page: ${escapeHtml(source.page)}`;
                    }
                    html += '</div>';
                    
                    // Show preview of source text (render markdown)
                    const previewText = source.text || source.chunk_text || '';
                    if (previewText) {
                        const truncated = previewText.length > 200 
                            ? previewText.substring(0, 200) + '...' 
                            : previewText;
                        // Render markdown instead of escaping HTML
                        const renderedMarkdown = marked.parse(truncated);
                        html += `<div class="source-preview markdown-content">${renderedMarkdown}</div>`;
                    }
                    
                    html += '</div>';
                });
                
                html += '</div>';
            }
            
            // Display similar drugs if available
            if (data.similar_drugs && data.similar_drugs.length > 0) {
                html += '<div class="similar-drugs-container">';
                html += '<h3>Sambærileg lyf (með sama virka innihaldsefni):</h3>';
                html += '<ul class="similar-drugs-list">';
                data.similar_drugs.forEach((drug) => {
                    html += `<li>${escapeHtml(drug)}</li>`;
                });
                html += '</ul>';
                html += '</div>';
            }
            
            resultDiv.innerHTML = html;
            
            // Clear question input after successful response (keep drug-id for follow-up)
            document.getElementById('question').value = '';
        } else {
            // Error response
            resultDiv.innerHTML = `<div class="error-message show">Error: ${escapeHtml(data.detail || data.error || 'Unknown error')}</div>`;
        }
    } catch (error) {
        // Stop rotating status messages on error
        clearInterval(statusInterval);
        resultDiv.innerHTML = `<div class="error-message show">Request failed: ${escapeHtml(error.message)}</div>`;
    }
}

// Clear session and reset form
async function clearSession() {
    try {
        const response = await fetch('/api/tools/clear_session', {
            method: 'POST',
            headers: {
                'Authorization': 'Basic ' + btoa(`${credentials.username}:${credentials.password}`),
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                session_id: 'web_session'
            })
        });
        
        if (response.ok) {
            // Clear form inputs
            document.getElementById('question').value = '';
            document.getElementById('drug-id').value = '';
            
            // Hide result container
            document.getElementById('result-container').style.display = 'none';
            
            // Optional: Show success message briefly
            const warmupStatus = document.getElementById('warmup-status');
            warmupStatus.style.display = 'block';
            warmupStatus.className = 'status-message success';
            warmupStatus.textContent = 'Samtalið hefur verið endurstillt';
            setTimeout(() => {
                warmupStatus.style.display = 'none';
            }, 2000);
        } else {
            const errorData = await response.json();
            const warmupStatus = document.getElementById('warmup-status');
            warmupStatus.style.display = 'block';
            warmupStatus.className = 'status-message error';
            warmupStatus.textContent = `Villa við að endurstilla: ${errorData.detail || 'Unknown error'}`;
        }
    } catch (error) {
        const warmupStatus = document.getElementById('warmup-status');
        warmupStatus.style.display = 'block';
        warmupStatus.className = 'status-message error';
        warmupStatus.textContent = `Villa: ${error.message}`;
    }
}

// Utility function to escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

