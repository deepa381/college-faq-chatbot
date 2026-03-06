// src/api/chatApi.js
// Axios wrapper for the /api/chat POST endpoint.
// Exports sendMessage(question) → Promise<{ answer, matched_question, category, confidence }>
// Module: API Layer

import axios from 'axios';

const BASE_URL = '/api';

/**
 * Send a user message to the backend and return the matched FAQ result.
 * @param {string} message - The user's question.
 * @param {string} [sessionId] - Optional session ID for conversation memory.
 * @returns {Promise<object>} - { answer, retrieved_entries, model_used, success, session_id }
 */
export async function sendMessage(message, sessionId) {
  const payload = { message };
  if (sessionId) {
    payload.session_id = sessionId;
  }
  const response = await axios.post(`${BASE_URL}/chat`, payload);
  return response.data;
}
