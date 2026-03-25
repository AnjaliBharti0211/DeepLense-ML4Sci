const API_URL = "http://127.0.0.1:8000/api/chat";

let sessionId = null;

const chatHistory = document.getElementById("chat-history");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const configDisplay = document.getElementById("config-display");
const imageGallery = document.getElementById("image-gallery");

function parseMarkdown(text) {
    let parsed = text;
    // Bold
    parsed = parsed.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    parsed = parsed.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Inline Code
    parsed = parsed.replace(/`(.*?)`/g, '<code>$1</code>');
    // Links
    parsed = parsed.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>');
    // Newlines to BR
    return parsed; // since we use white-space: pre-wrap in CSS, we don't strictly need <br>
}

function addMessage(text, isUser = false) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${isUser ? 'user-message' : 'agent-message'}`;
    
    const content = document.createElement("div");
    content.className = "message-content";
    content.innerHTML = parseMarkdown(text);
    
    msgDiv.appendChild(content);
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function addTypingIndicator() {
    const typing = document.createElement("div");
    typing.className = "typing-indicator";
    typing.id = "typing-indicator";
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("div");
        dot.className = "typing-dot";
        typing.appendChild(dot);
    }
    
    chatHistory.appendChild(typing);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function removeTypingIndicator() {
    const typing = document.getElementById("typing-indicator");
    if (typing) {
        typing.remove();
    }
}

function updateConfig(config) {
    if (!config) return;
    configDisplay.textContent = JSON.stringify(config, null, 2);
}

function addImages(images) {
    if (!images || images.length === 0) return;
    
    const placeholder = document.querySelector(".placeholder-text");
    if (placeholder) {
        placeholder.remove();
    }
    
    images.forEach(img => {
        const imgEl = document.createElement("img");
        imgEl.className = "sim-image";
        imgEl.src = `data:image/png;base64,${img.base64_png}`;
        imgEl.title = `Sim ID: ${img.simulation_id} | Index: ${img.index}`;
        imageGallery.appendChild(imgEl);
    });
}

async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;
    
    // UI Updates
    chatInput.value = "";
    sendBtn.disabled = true;
    chatInput.disabled = true;
    addMessage(text, true);
    addTypingIndicator();
    
    try {
        const payload = {
            message: text,
            session_id: sessionId
        };
        
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        sessionId = data.session_id;
        removeTypingIndicator();
        addMessage(data.reply, false);
        
        if (data.config) {
            updateConfig(data.config);
        }
        
        if (data.images && data.images.length > 0) {
            addImages(data.images);
        }
        
    } catch (e) {
        console.error("Error communicating with agent", e);
        removeTypingIndicator();
        addMessage("⚠️ I encountered an error communicating with the backend. Please check if the FastAPI server is running. See console for details.", false);
    } finally {
        sendBtn.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
    }
}

sendBtn.addEventListener("click", sendMessage);

chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") {
        sendMessage();
    }
});
