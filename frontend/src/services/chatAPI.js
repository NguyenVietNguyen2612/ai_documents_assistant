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

export async function sendMessageStream(question, history = [], onEvent, onError, onComplete) {
    try {
        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                question: question,
                history: history,
            }),
        });

        if (!response.ok) {
            throw new Error(`Chat stream failed with status: ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop(); // giữ lại phần chưa đủ dòng

            for (const line of lines) {
                const trimmed = line.trim();
                if (trimmed.startsWith("data:")) {
                    const jsonStr = trimmed.slice(5).trim();
                    if (!jsonStr) continue;
                    try {
                        const event = JSON.parse(jsonStr);
                        if (onEvent) onEvent(event);
                    } catch (err) {
                        console.error("Error parsing SSE data:", err, jsonStr);
                    }
                }
            }
        }

        if (onComplete) onComplete();
    } catch (err) {
        if (onError) onError(err);
        else console.error("Stream error:", err);
    }
}