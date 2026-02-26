// src/components/ChatWindow.jsx
// Main chat UI container.
// Renders the scrollable message list and the InputBar.
// Consumes the useChat() hook for state and actions.
// Module: UI Components

import React, { useEffect, useRef } from 'react';
import { useChat } from '../hooks/useChat';
import { MessageBubble } from './MessageBubble';
import { InputBar } from './InputBar';

export function ChatWindow() {
    const { messages, loading, sendMessage } = useChat();
    const bottomRef = useRef(null);

    // Auto-scroll to the latest message
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, loading]);

    return (
        <div className="chat-window">
            {/* Message list */}
            <div className="chat-window__messages" aria-live="polite" aria-label="Chat messages">
                {messages.map((msg) => (
                    <MessageBubble key={msg.id} message={msg} />
                ))}

                {/* Typing indicator */}
                {loading && (
                    <div className="message-row message-row--bot">
                        <div className="avatar avatar--bot"><span>🎓</span></div>
                        <div className="bubble bubble--bot bubble--typing">
                            <span className="dot" />
                            <span className="dot" />
                            <span className="dot" />
                        </div>
                    </div>
                )}

                <div ref={bottomRef} />
            </div>

            {/* Input area */}
            <div className="chat-window__footer">
                <InputBar onSend={sendMessage} disabled={loading} />
            </div>
        </div>
    );
}
