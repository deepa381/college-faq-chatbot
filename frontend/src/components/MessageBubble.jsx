// src/components/MessageBubble.jsx
// Renders a single chat message bubble (user or bot).
// Module: UI Components

import React from 'react';

/**
 * @param {{ message: { role: string, text: string, category?: string, confidence?: number, matched?: boolean } }} props
 */
export function MessageBubble({ message }) {
    const isUser = message.role === 'user';

    return (
        <div className={`message-row ${isUser ? 'message-row--user' : 'message-row--bot'}`}>
            {!isUser && (
                <div className="avatar avatar--bot" aria-label="Bot">
                    <span>🎓</span>
                </div>
            )}

            <div className={`bubble ${isUser ? 'bubble--user' : 'bubble--bot'}`}>
                <p className="bubble__text">{message.text}</p>
                {!isUser && message.category && (
                    <span className="bubble__tag">{message.category}</span>
                )}
            </div>

            {isUser && (
                <div className="avatar avatar--user" aria-label="You">
                    <span>👤</span>
                </div>
            )}
        </div>
    );
}
