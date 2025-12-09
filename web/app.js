// Global state
let credentials = {
    username: '',
    password: ''
};

// Store citations for tooltip display
let currentCitations = [];

/**
 * Find the index of a source in the sources array by matching drug_id and section_number.
 * 
 * @param {Array} sources - Array of source objects from the API
 * @param {string} drugId - Drug ID from the citation
 * @param {string} sectionNumber - Section number from the citation
 * @returns {number} - 1-based index of the matching source, or -1 if not found
 */
function findSourceIndex(sources, drugId, sectionNumber) {
    if (!sources || !Array.isArray(sources)) return -1;
    
    const index = sources.findIndex(s => {
        const sourceDrug = (s.drug_id || '').toLowerCase();
        const sourceSection = (s.section_number || '').toString().trim();
        const citationDrug = drugId.toLowerCase();
        const citationSection = sectionNumber.trim();
        
        // Match drug_id (allow partial match for variations like _SmPC suffix)
        const drugMatch = sourceDrug.includes(citationDrug) || citationDrug.includes(sourceDrug);
        // Match section number exactly
        const sectionMatch = sourceSection === citationSection;
        
        return drugMatch && sectionMatch;
    });
    
    // Return 1-based index (for display), or -1 if not found
    return index >= 0 ? index + 1 : -1;
}

/**
 * Parse citation tags from answer text and replace with superscript numbers.
 * Citation format: [drug_id, kafli section_number: section_title]
 * 
 * Links citations to actual source indices from the sources array.
 * 
 * @param {string} html - The HTML content containing citation tags
 * @param {Array} sources - Array of source objects from the API response
 * @returns {object} - Object with parsed HTML and citations array
 */
function parseCitations(html, sources) {
    const citations = [];
    // Match citation pattern: [drug_id, kafli section_number: section_title]
    const citationRegex = /\[([^\],]+),\s*kafli\s+([^:]+):\s*([^\]]+)\]/gi;
    
    let fallbackIndex = 0;
    const seenCitations = new Map(); // Track unique citations
    
    // First pass: collect unique citations and find their source indices
    const tempHtml = html;
    let match;
    while ((match = citationRegex.exec(tempHtml)) !== null) {
        const drugId = match[1].trim();
        const sectionNumber = match[2].trim();
        const sectionTitle = match[3].trim();
        const key = `${drugId}-${sectionNumber}`;
        
        if (!seenCitations.has(key)) {
            // Find the actual source index from the sources array
            let sourceIndex = findSourceIndex(sources, drugId, sectionNumber);
            
            // Fallback to sequential numbering if source not found
            if (sourceIndex === -1) {
                fallbackIndex++;
                sourceIndex = fallbackIndex;
                console.warn(`Citation not found in sources: ${drugId}, kafli ${sectionNumber}`);
            }
            
            seenCitations.set(key, {
                index: sourceIndex,
                drugId: drugId,
                sectionNumber: sectionNumber,
                sectionTitle: sectionTitle
            });
            citations.push({
                index: sourceIndex,
                drugId: drugId,
                sectionNumber: sectionNumber,
                sectionTitle: sectionTitle
            });
        }
    }
    
    // Second pass: replace citations with superscript numbers linked to actual sources
    const parsedHtml = html.replace(citationRegex, (match, drugId, sectionNumber, sectionTitle) => {
        const key = `${drugId.trim()}-${sectionNumber.trim()}`;
        const citation = seenCitations.get(key);
        if (citation) {
            const superscriptNum = getSuperscriptNumber(citation.index);
            return `<sup class="citation-link" data-index="${citation.index}" data-drug="${escapeHtml(citation.drugId)}" data-section="${escapeHtml(citation.sectionNumber)}" data-title="${escapeHtml(citation.sectionTitle)}">${superscriptNum}</sup>`;
        }
        return match;
    });
    
    return { html: parsedHtml, citations: citations };
}

/**
 * Convert a number to superscript Unicode characters.
 * 
 * @param {number} num - The number to convert
 * @returns {string} - Superscript representation
 */
function getSuperscriptNumber(num) {
    const superscripts = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    return String(num).split('').map(d => superscripts[parseInt(d)]).join('');
}

/**
 * Initialize citation tooltips for hover behavior.
 */
function initCitationTooltips() {
    document.querySelectorAll('.citation-link').forEach(el => {
        el.addEventListener('mouseenter', showCitationTooltip);
        el.addEventListener('mouseleave', hideCitationTooltip);
        el.addEventListener('click', scrollToSource);
    });
}

/**
 * Show tooltip on citation hover.
 */
function showCitationTooltip(event) {
    const el = event.target;
    const drugId = el.dataset.drug;
    const sectionNumber = el.dataset.section;
    const sectionTitle = el.dataset.title;
    
    // Remove any existing tooltip
    hideCitationTooltip();
    
    // Create tooltip
    const tooltip = document.createElement('div');
    tooltip.className = 'citation-tooltip';
    tooltip.innerHTML = `<strong>${drugId}</strong><br>Kafli ${sectionNumber}: ${sectionTitle}`;
    
    // Position tooltip
    const rect = el.getBoundingClientRect();
    tooltip.style.left = `${rect.left + window.scrollX}px`;
    tooltip.style.top = `${rect.bottom + window.scrollY + 5}px`;
    
    document.body.appendChild(tooltip);
}

/**
 * Hide citation tooltip.
 */
function hideCitationTooltip() {
    const existing = document.querySelector('.citation-tooltip');
    if (existing) {
        existing.remove();
    }
}

/**
 * Scroll to and highlight the corresponding source when citation is clicked.
 */
function scrollToSource(event) {
    const index = event.target.dataset.index;
    const sourceItem = document.querySelector(`.source-item[data-index="${index}"]`);
    if (sourceItem) {
        // Open the sources drawer if closed
        const drawer = document.querySelector('.sources-drawer');
        if (drawer && !drawer.classList.contains('open')) {
            toggleSourcesDrawer();
        }
        // Scroll to source
        sourceItem.scrollIntoView({ behavior: 'smooth', block: 'center' });
        sourceItem.classList.add('highlight');
        setTimeout(() => sourceItem.classList.remove('highlight'), 2000);
    }
}

/**
 * Toggle the sources drawer open/closed.
 */
function toggleSourcesDrawer() {
    const drawer = document.querySelector('.sources-drawer');
    const button = document.querySelector('.btn-view-sources');
    if (drawer) {
        drawer.classList.toggle('open');
        if (button) {
            button.textContent = drawer.classList.contains('open') ? 'Fela heimildir' : 'Skoða heimildir';
        }
    }
}

/**
 * Handle suggested query chip click.
 */
function handleChipClick(query) {
    const questionInput = document.getElementById('question');
    if (questionInput) {
        questionInput.value = query;
        questionInput.focus();
    }
}

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
            errorDiv.textContent = 'Rangt notandanafn eða lykilorð';
            errorDiv.classList.add('show');
        }
    } catch (error) {
        errorDiv.textContent = `Villa við innskráningu: ${error.message}`;
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
    warmupStatus.textContent = 'Ræsi kerfi...';
    
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
                warmupStatus.textContent = 'Kerfi tilbúið (þegar ræst)';
            } else {
                warmupStatus.textContent = 'Kerfi ræst og tilbúið!';
            }
            // Hide warmup status after 3 seconds
            setTimeout(() => {
                warmupStatus.style.display = 'none';
            }, 3000);
        } else {
            const errorData = await response.json();
            warmupStatus.className = 'status-message error';
            warmupStatus.textContent = `Villa við ræsingu: ${errorData.detail || 'Óþekkt villa'}`;
        }
    } catch (error) {
        warmupStatus.className = 'status-message error';
        warmupStatus.textContent = `Villa: ${error.message}`;
    }
}

// Status messages that reflect the RAG pipeline steps with progressive detail
const queryStatusMessages = [
    { text: 'Sæki samtalssögu...', icon: '📚' },
    { text: 'Greini fyrirspurn og finn lyf...', icon: '🔍' },
    { text: 'Umbreyti leitarorðum...', icon: '✏️' },
    { text: 'Leita í SmPC gagnagrunni...', icon: '🗄️' },
    { text: 'Raða skjölum eftir mikilvægi...', icon: '📊' },
    { text: 'Les viðeigandi kafla úr SmPC skjölum...', icon: '📖' },
    { text: 'Smíða svar með tilvísunum...', icon: '💬' },
    { text: 'Finn samheitalyf...', icon: '💊' },
    { text: 'Staðfesti heimildir...', icon: '✅' }
    
];

/**
 * Show rotating status messages during query processing.
 * Shows progressive steps to indicate system is working.
 * 
 * @param {HTMLElement} resultDiv - The div element to update with status messages
 * @returns {number} - Interval ID that can be used to clear the interval
 */
function showQueryStatus(resultDiv) {
    let messageIndex = 0;
    
    // Show initial message with step indicator
    const renderStatus = (index) => {
        const msg = queryStatusMessages[index];
        const stepText = `Skref ${index + 1}/${queryStatusMessages.length}`;
        resultDiv.innerHTML = `
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-step">${stepText}</div>
                <div class="loading-message">${msg.icon} ${msg.text}</div>
            </div>
        `;
    };
    
    renderStatus(0);
    
    // Rotate through messages every 4.5 seconds (faster progression). FINAL message should persist indefinitely.
    const intervalId = setInterval(() => {
        if (messageIndex === queryStatusMessages.length - 1) {
            // Final message should persist indefinitely
            renderStatus(messageIndex);
        } else {
            messageIndex = (messageIndex + 1) % queryStatusMessages.length;
            renderStatus(messageIndex);
        }
        renderStatus(messageIndex);
    }, 4500);
    
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
            
            // Display answer with citation parsing and markdown rendering
            html += '<div class="answer-container">';
            html += '<h3>Svar:</h3>';
            // Step 1: Parse citations FIRST from raw answer text (before markdown)
            // This prevents marked.js from altering the [drug_id, kafli X: title] pattern
            const citationResult = parseCitations(data.answer, data.sources || []);
            currentCitations = citationResult.citations;
            // Step 2: Render markdown (the <sup> tags from citations are preserved)
            let renderedMarkdown = marked.parse(citationResult.html);
            html += `<div class="markdown-content">${renderedMarkdown}</div>`;
            html += '</div>';
            
            // Display sources in collapsible drawer if available
            if (data.sources && data.sources.length > 0) {
                // Add button to toggle sources drawer
                html += '<button class="btn btn-view-sources" onclick="toggleSourcesDrawer()">Skoða heimildir</button>';
                
                html += '<div class="sources-drawer">';
                html += '<div class="sources-container">';
                html += `<h3>Heimildir (${data.sources.length})</h3>`;
                
                data.sources.forEach((source, index) => {
                    const sourceIndex = index + 1;
                    html += `<div class="source-item" data-index="${sourceIndex}">`;
                    html += '<div class="source-header">';
                    html += `<span class="source-number">${getSuperscriptNumber(sourceIndex)}</span>`;
                    html += `<strong>${escapeHtml(source.drug_id || 'Óþekkt')}</strong>`;
                    html += '</div>';
                    html += '<div class="source-meta">';
                    html += `<span class="source-section">Kafli ${escapeHtml(source.section_number || 'Óþekkt')}: ${escapeHtml(source.section_title || 'Óþekkt')}</span>`;
                    if (source.page && source.page !== 'Unknown') {
                        html += `<span class="source-page"> | Síða: ${escapeHtml(source.page)}</span>`;
                    }
                    html += '</div>';
                    
                    // Show preview of source text (render markdown) in expandable section
                    const previewText = source.text || source.chunk_text || '';
                    if (previewText) {
                        const truncated = previewText.length > 500 
                            ? previewText.substring(0, 500) + '...' 
                            : previewText;
                        // Render markdown instead of escaping HTML
                        const renderedPreview = marked.parse(truncated);
                        html += `<div class="source-preview markdown-content">${renderedPreview}</div>`;
                    }
                    
                    html += '</div>';
                });
                
                html += '</div>';
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
            
            // Initialize citation tooltips after rendering
            initCitationTooltips();
            
            // Clear question input after successful response (keep drug-id for follow-up)
            document.getElementById('question').value = '';
        } else {
            // Error response
            resultDiv.innerHTML = `<div class="error-message show">Villa: ${escapeHtml(data.detail || data.error || 'Óþekkt villa')}</div>`;
        }
    } catch (error) {
        // Stop rotating status messages on error
        clearInterval(statusInterval);
        resultDiv.innerHTML = `<div class="error-message show">Beiðni mistókst: ${escapeHtml(error.message)}</div>`;
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

