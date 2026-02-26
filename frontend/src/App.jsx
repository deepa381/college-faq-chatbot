// src/App.jsx
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api/chat';
import './App.css';

export default function App() {
    const [messages, setMessages] = useState([
        { id: 0, role: 'bot', text: "Hi! I'm the KARE FAQ Assistant. Ask me anything about Kalasalingam University." }
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const [sending, setSending] = useState(false);  // Prevents rapid double-clicks
    const bottomRef = useRef(null);
    const inputRef = useRef(null);

    // Auto-scroll to latest message
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    // Re-focus input after bot responds
    useEffect(() => {
        if (!loading) inputRef.current?.focus();
    }, [loading]);

    const sendMessage = async () => {
        const text = input.trim();

        // ✅ Handle empty message input
        if (!text) return;

        // ✅ Prevent multiple rapid submissions
        if (loading || sending) return;
        setSending(true);

        // Add user message
        const userMsg = { id: Date.now(), role: 'user', text };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setLoading(true);

        try {
            // POST { message: userInput } to backend with timeout
            const { data } = await axios.post(API_URL, { message: text }, {
                timeout: 30000  // 30s timeout — Gemini calls may take a few seconds
            });

            // Extract category from retrieved entries if available
            const category = data.retrieved_entries?.length > 0
                ? data.retrieved_entries[0].category
                : null;

            const botMsg = {
                id: Date.now() + 1,
                role: 'bot',
                text: data.answer,
                category,
                isError: !data.success,
            };
            setMessages(prev => [...prev, botMsg]);

        } catch (err) {
            // ✅ Handle server downtime + API errors with fallback UI
            let fallback = '⚠️ Something went wrong. Please try again.';

            if (err.code === 'ECONNABORTED') {
                // Request timed out
                fallback = '⏱️ The server took too long to respond. Please try again later.';
            } else if (err.response) {
                // Server responded with non-2xx status
                const status = err.response.status;
                fallback = `⚠️ Server error (${status}): ${err.response.data?.error || 'Unexpected error.'}`;
            } else if (err.request) {
                // No response (backend down / CORS)
                fallback = '🔌 Cannot reach the server. Make sure the backend is running on http://localhost:5000';
            }

            setMessages(prev => [
                ...prev,
                { id: Date.now() + 1, role: 'bot', text: fallback, isError: true }
            ]);
        } finally {
            setLoading(false);
            setSending(false);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    };

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <span className="header-icon">🎓</span>
                <div>
                    <div className="header-title">KARE FAQ Assistant</div>
                    <div className="header-sub">Kalasalingam Academy of Research and Education</div>
                </div>
            </header>

            {/* Message list */}
            <main className="messages">
                {messages.map(msg => (
                    <div key={msg.id} className={`row row-${msg.role}`}>
                        <div className={`bubble bubble-${msg.role} ${msg.isError ? 'bubble-error' : ''}`}>
                            <p>{msg.text}</p>
                            {msg.category && (
                                <span className="category-tag">{msg.category}</span>
                            )}
                        </div>
                    </div>
                ))}

                {/* ✅ Loading spinner while waiting for response */}
                {loading && (
                    <div className="row row-bot">
                        <div className="bubble bubble-bot typing">
                            <div className="spinner" />
                            <span className="typing-text">Thinking…</span>
                        </div>
                    </div>
                )}

                <div ref={bottomRef} />
            </main>

            {/* Input bar */}
            <footer className="footer">
                <textarea
                    ref={inputRef}
                    id="chat-input"
                    className="input"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about admissions, fees, placements…"
                    disabled={loading}
                    rows={1}
                />
                <button
                    id="send-btn"
                    className="send-btn"
                    onClick={sendMessage}
                    disabled={loading || sending || !input.trim()}
                >
                    {loading ? (
                        <div className="btn-spinner" />
                    ) : (
                        'Send'
                    )}
                </button>
            </footer>
        </div>
    );
}
