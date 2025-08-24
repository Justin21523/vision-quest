import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage, ChatRequest, ChatResponse } from '../services/types';
import { chatCompletion } from '../services/api';

interface ChatCardProps {
  className?: string;
}

export const ChatCard: React.FC<ChatCardProps> = ({ className = '' }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'system', content: '你是一個友善且有幫助的AI助手。', timestamp: '' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState('qwen');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString()
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const request: ChatRequest = {
        messages: newMessages,
        model: selectedModel,
        max_tokens: 1024,
        temperature: 0.7,
        stream: false
      };

      const response: ChatResponse = await chatCompletion(request);

      setMessages(prev => [...prev, {
        ...response.message,
        timestamp: new Date().toISOString()
      }]);

    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '抱歉，發生錯誤。請稍後重試。',
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className={`chat-card bg-white rounded-lg shadow-md p-6 ${className}`}>
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-800">💬 智能對話</h2>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          className="px-3 py-1 border rounded-md text-sm"
        >
          <option value="qwen">Qwen2-7B</option>
          <option value="llama">Llama-3.1-8B</option>
        </select>
      </div>

      {/* 對話歷史 */}
      <div className="chat-history h-96 overflow-y-auto border rounded-lg p-4 mb-4 bg-gray-50">
        {messages.filter(msg => msg.role !== 'system').map((message, index) => (
          <div key={index} className={`mb-4 ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
            <div className={`inline-block max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
              message.role === 'user'
                ? 'bg-blue-500 text-white'
                : 'bg-white border text-gray-800'
            }`}>
              <div className="whitespace-pre-wrap">{message.content}</div>
              <div className="text-xs opacity-70 mt-1">
                {message.timestamp ? new Date(message.timestamp).toLocaleTimeString() : ''}
              </div>
            </div>
          </div>
        ))}
        {loading && (
          <div className="text-left mb-4">
            <div className="inline-block bg-gray-200 px-4 py-2 rounded-lg">
              <div className="flex items-center space-x-2">
                <div className="typing-indicator">
                  <span></span><span></span><span></span>
                </div>
                <span className="text-gray-600">思考中...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 輸入框 */}
      <div className="flex space-x-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="輸入您的問題... (Enter 發送，Shift+Enter 換行)"
          className="flex-1 p-3 border rounded-lg resize-none"
          rows={2}
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
        >
          {loading ? '發送中...' : '發送'}
        </button>
      </div>
    </div>
  );
};