document.addEventListener("DOMContentLoaded", () => {
    const setupBtn = document.getElementById("setupKeyBtn");
    const modelInput = document.getElementById("modelName");
    const apiKeyInput = document.getElementById("customApiKey");
    const status = document.getElementById("status");

    setupBtn.addEventListener("click", async () => {
        const model = modelInput.value.trim();
        const apiKey = apiKeyInput.value.trim();
        if (!model || !apiKey) {
            status.textContent = "Please fill in both fields.";
            return;
        }

        // For demo, we just display success
        status.textContent = `Custom API key for ${model} set! It may take 1hr-7days depending on model size.`;
        // Later: call backend endpoint to save key
    });
});