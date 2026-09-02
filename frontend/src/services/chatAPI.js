const API_BASE_URL = "http://localhost:8001";

export async function sendMessage(question, history = []) {
    const response = await fetch(
        `${API_BASE_URL}/chat`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json",
            },

            body: JSON.stringify({
                question: question,
                history: history,
            }),
        }
    );

    if (!response.ok) {
        throw new Error(
            `Chat API failed: ${response.status}`
        );
    }

    return await response.json();
}