const API_BASE_URL = "http://localhost:8001";

export async function sendMessage(question, history = [], sessionId = "default", checkpointId = null, documentFilter = null) {
    const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            question: question,
            history: history,
            session_id: sessionId,
            checkpoint_id: checkpointId,
            document_filter: documentFilter,
        }),
    });

    if (!response.ok) {
        throw new Error(`Chat API failed: ${response.status}`);
    }

    return await response.json();
}

export async function sendMessageStream({
    question,
    history = [],
    sessionId = "default",
    checkpointId = null,
    rollbackFromMessageId = null,
    documentFilter = null,
    onEvent,
    onError,
    onComplete
}) {
    try {
        const response = await fetch(`${API_BASE_URL}/chat/stream`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                question: question,
                history: history,
                session_id: sessionId,
                checkpoint_id: checkpointId,
                rollback_from_message_id: rollbackFromMessageId,
                document_filter: documentFilter,
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
            buffer = lines.pop(); // Giữ lại phần chưa hoàn chỉnh

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

// --- Session APIs ---

export async function getSessions() {
    const res = await fetch(`${API_BASE_URL}/chat/sessions`);
    if (!res.ok) throw new Error("Failed to load sessions");
    return await res.json();
}

export async function createSession(title = "Cuộc trò chuyện mới") {
    const res = await fetch(`${API_BASE_URL}/chat/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
    });
    if (!res.ok) throw new Error("Failed to create session");
    return await res.json();
}

export async function deleteSession(sessionId) {
    const res = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}`, {
        method: "DELETE",
    });
    if (!res.ok) throw new Error("Failed to delete session");
    return await res.json();
}

export async function getSessionMessages(sessionId) {
    const res = await fetch(`${API_BASE_URL}/chat/sessions/${sessionId}/messages`);
    if (!res.ok) throw new Error("Failed to load session messages");
    return await res.json();
}

// --- Checkpoint & Rollback APIs ---

export async function getCheckpoints(sessionId) {
    const res = await fetch(`${API_BASE_URL}/chat/checkpoints?session_id=${encodeURIComponent(sessionId)}`);
    if (!res.ok) throw new Error("Failed to load checkpoints");
    return await res.json();
}

export async function rollbackChat(sessionId, checkpointId, messageId = null) {
    const res = await fetch(`${API_BASE_URL}/chat/rollback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            session_id: sessionId,
            checkpoint_id: checkpointId,
            message_id: messageId,
        }),
    });
    if (!res.ok) throw new Error("Failed to rollback");
    return await res.json();
}