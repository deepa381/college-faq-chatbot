// src/hooks/useChat.js
// Custom React hook encapsulating all chat state logic.
// Manages messages[], loading state, and the sendMessage action via chatApi.
// Module: Custom Hooks

import { useState, useCallback } from 'react';
import { sendMessage as apiSendMessage } from '../api/chatApi';

/**
 * useChat hook — provides messages, loading state, and sendMessage action.
 */
export function useChat() {
    const [messages, setMessages] = useState([
        {
            id: 'welcome',
            role: 'bot',
            text: "Hi! I'm the KARE FAQ Assistant 👋 Ask me anything about Kalasalingam University — admissions, fees, placements, hostel, and more!",
            timestamp: new Date(),
        },
    ]);
    const [loading, setLoading] = useState(false);

    const sendMessage = useCallback(async (text) => {
        const trimmed = text.trim();
        if (!trimmed) return;

        const userMsg = {
            id: `user-${Date.now()}`,
            role: 'user',
            text: trimmed,
            timestamp: new Date(),
        };

        setMessages((prev) => [...prev, userMsg]);
        setLoading(true);

        try {
            const result = await apiSendMessage(trimmed);
            const botMsg = {
                id: `bot-${Date.now()}`,
                role: 'bot',
                text: result.answer,
                category: result.category,
                confidence: result.confidence,
                matched: result.matched,
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, botMsg]);
        } catch (err) {
            const errMsg = {
                id: `err-${Date.now()}`,
                role: 'bot',
                text: '⚠️ Sorry, I could not reach the server. Please make sure the backend is running.',
                timestamp: new Date(),
            };
            setMessages((prev) => [...prev, errMsg]);
        } finally {
            setLoading(false);
        }
    }, []);

    return { messages, loading, sendMessage };
}
