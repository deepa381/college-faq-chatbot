// src/components/InputBar.jsx
// Text input bar for composing and sending chat messages.
// Module: UI Components

import React, { useState } from 'react';

/**
 * @param {{ onSend: (text: string) => void, disabled: boolean }} props
 */
export function InputBar({ onSend, disabled }) {
    const [value, setValue] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (!value.trim() || disabled) return;
        onSend(value);
        setValue('');
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            handleSubmit(e);
        }
    };

    return (
        <form className="input-bar" onSubmit={handleSubmit} aria-label="Chat input">
            <textarea
                id="chat-input"
                className="input-bar__textarea"
                value={value}
                onChange={(e) => setValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask about admissions, fees, placements…"
                disabled={disabled}
                rows={1}
                aria-label="Type your question"
            />
            <button
                id="send-btn"
                type="submit"
                className="input-bar__send"
                disabled={disabled || !value.trim()}
                aria-label="Send message"
            >
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
                    <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                </svg>
            </button>
        </form>
    );
}
