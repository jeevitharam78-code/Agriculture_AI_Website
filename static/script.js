// Get HTML elements
const inputBox = document.querySelector("input");
const sendButton = document.querySelectorAll("button")[0];
const clearButton = document.querySelectorAll("button")[1];
const chatHistory = document.querySelector("h3").nextElementSibling;

// Send button
sendButton.addEventListener("click", async function () {

    const question = inputBox.value.trim();

    if (question === "") {
        alert("Please enter a question.");
        return;
    }

    // Show user's question
    chatHistory.innerHTML += `
    <div class="user-message">
        <b>👨‍🌾 Farmer:</b><br>${question}
    </div>
`;
chatHistory.scrollTop = chatHistory.scrollHeight;
    inputBox.value = "";

    try {

        chatHistory.innerHTML += `
    <div class="loading" id="loading">
        🌾 Agriculture AI is thinking...
    </div>
`;const response = await fetch("/ask", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                question: question
            })
        });

        const data = await response.json();
        document.getElementById("loading").remove();

        // Show AI answer
        chatHistory.innerHTML += `
    <div class="ai-message">
        <b>🌾 Agriculture AI:</b><br>${data.answer}
    </div>
`;
chatHistory.scrollTop = chatHistory.scrollHeight;
    }
    catch (error) {

        chatHistory.innerHTML += `
            <p style="color:red;">
            Error connecting to AI.
            </p>
        `;

    }

});

// Clear Chat
clearButton.addEventListener("click", function () {

    chatHistory.innerHTML = "<p>No messages yet.</p>";

});
// Press Enter to send the question
inputBox.addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        sendButton.click();
    }
});