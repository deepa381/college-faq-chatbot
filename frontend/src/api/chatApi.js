// src/api/chatApi.js
// Axios wrapper for the /api/chat POST endpoint.
// Exports sendMessage(question) → Promise<{ answer, matched_question, category, confidence }>
// Module: API Layer

import axios from 'axios';

const BASE_URL = '/api';

/**
 * Send a user message to the backend and return the matched FAQ result.
 * @param {string} message - The user's question.
 * @returns {Promise<object>} - { answer, matched_question, category, confidence, matched }
 */
export async function sendMessage(message) {
  const response = await axios.post(`${BASE_URL}/chat`, { message });
  return response.data;
}
