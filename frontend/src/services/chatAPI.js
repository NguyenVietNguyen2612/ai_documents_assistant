const API_BASE_URL = "http://localhost:8000";

export async function sendMessage(question) {
    const response = await fetch(
        `${API_BASE_URL}/chat`,
        {
            method: "POST",

            headers: {
                "Content-Type": "application/json",
            },

            body: JSON.stringify({
                question: question,
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