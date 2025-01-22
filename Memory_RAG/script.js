document.addEventListener("DOMContentLoaded", function () {
    const chatMessages = document.getElementById("chat-messages");
    const messageInput = document.getElementById("messageInput");
    const loadingIndicator = document.querySelector(".loading-indicator");

    function appendMessage(content, sender) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);
        
        // Parse markdown if it's a bot message
        if (sender === "bot") {
            messageDiv.innerHTML = marked.parse(content);
        } else {
            messageDiv.textContent = content;
        }
        
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showLoading() {
        loadingIndicator.classList.remove("hidden");
        scrollToBottom();
    }

    function hideLoading() {
        loadingIndicator.classList.add("hidden");
    }

    async function sendMessage() {
        const userMessage = messageInput.value.trim();
        if (!userMessage) return;

        appendMessage(userMessage, "user");
        messageInput.value = "";
        showLoading();

        try {
            const response = await fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userMessage }),
            });
            const data = await response.json();
            hideLoading();
            appendMessage(data.reply, "bot");
        } catch (error) {
            hideLoading();
            appendMessage("Error: Unable to fetch response from server.", "bot");
        }
    }

    // Handle Enter key press
    messageInput.addEventListener("keypress", function(event) {
        if (event.key === "Enter") {
            event.preventDefault();
            sendMessage();
        }
    });

    window.sendMessage = sendMessage;
});
