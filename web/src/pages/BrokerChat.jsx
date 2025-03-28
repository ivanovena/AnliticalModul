import React, { useState, useRef, useEffect } from 'react';
import { usePortfolio } from '../contexts/PortfolioContext';
import { api } from '../services/api';

const BrokerChat = () => {
  const { portfolio } = usePortfolio();
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const messagesEndRef = useRef(null);
  
  // Opciones de prompt rápido
  const quickPrompts = [
    { id: 1, text: '¿Cómo está el mercado hoy?' },
    { id: 2, text: '¿Qué acciones recomiendas?' },
    { id: 3, text: '¿Cuál es tu análisis de AAPL?' },
    { id: 4, text: '¿Cuáles son tus recomendaciones de diversificación?' },
    { id: 5, text: '¿Cómo interpreto el indicador RSI?' }
  ];
  
  // Scroll al último mensaje
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  // Añadir mensaje de bienvenida inicial
  useEffect(() => {
    setMessages([{
      id: 'welcome',
      sender: 'assistant',
      content: '👋 ¡Hola! Soy tu asistente de trading IA. Puedo ayudarte con análisis de mercado, recomendaciones de inversión y resolver tus dudas sobre trading. ¿En qué puedo ayudarte hoy?',
      timestamp: new Date().toISOString()
    }]);
  }, []);
  
  // Procesar mensaje de entrada
  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (!inputMessage.trim()) return;
    
    // Añadir símbolo si está seleccionado
    const messageToSend = selectedSymbol 
      ? `[${selectedSymbol}] ${inputMessage}`
      : inputMessage;
    
    // Añadir mensaje del usuario a la conversación
    const userMessage = {
      id: Date.now().toString(),
      sender: 'user',
      content: messageToSend,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setLoading(true);
    
    try {
      const response = await api.sendChatMessage(messageToSend, conversationId);
      
      // Actualizar ID de conversación
      if (response.conversation_id) {
        setConversationId(response.conversation_id);
      }
      
      // Añadir respuesta del asistente
      const assistantMessage = {
        id: Date.now().toString() + '-response',
        sender: 'assistant',
        content: response.response,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error al enviar mensaje:', error);
      
      // Añadir mensaje de error
      const errorMessage = {
        id: Date.now().toString() + '-error',
        sender: 'assistant',
        content: 'Lo siento, ha ocurrido un error al procesar tu mensaje. Por favor, inténtalo de nuevo.',
        timestamp: new Date().toISOString(),
        isError: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };
  
  // Usar un prompt rápido
  const handleQuickPrompt = (prompt) => {
    setInputMessage(prompt);
  };
  
  // Formatear mensaje (básico)
  const formatMessage = (content) => {
    // Convertir enlaces a elementos <a>
    const urlRegex = /(https?:\/\/[^\s]+)/g;
    const withLinks = content.replace(urlRegex, url => `<a href="${url}" target="_blank" class="text-blue-600 hover:underline">${url}</a>`);
    
    // Formateo básico
    const withBasicFormatting = withLinks
      .replace(/\*\*(.*?)\*\*/g, '<span class="font-bold">$1</span>')  // Negrita
      .replace(/\*(.*?)\*/g, '<span class="italic">$1</span>')         // Cursiva
      .replace(/`(.*?)`/g, '<code class="bg-gray-100 px-1 rounded text-sm">$1</code>'); // Código
    
    // Convertir saltos de línea
    return withBasicFormatting.split('\n').map((line, i) => (
      <React.Fragment key={i}>
        <span dangerouslySetInnerHTML={{ __html: line }} />
        {i < withBasicFormatting.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));
  };
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
      {/* Panel principal del chat */}
      <div className="lg:col-span-3">
        <div className="bg-white rounded-lg shadow h-[calc(100vh-10rem)]">
          {/* Encabezado */}
          <div className="border-b p-4">
            <h2 className="text-lg font-semibold">Chat con Broker IA</h2>
          </div>
          
          {/* Área de mensajes */}
          <div className="p-4 h-[calc(100%-10rem)] overflow-y-auto">
            <div className="space-y-4">
              {messages.map(message => (
                <div
                  key={message.id}
                  className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-3 ${
                      message.sender === 'user'
                        ? 'bg-indigo-100 text-indigo-800'
                        : message.isError
                          ? 'bg-red-100 text-red-800'
                          : 'bg-gray-100 text-gray-800'
                    }`}
                  >
                    {formatMessage(message.content)}
                  </div>
                </div>
              ))}
              
              {/* Indicador de carga */}
              {loading && (
                <div className="flex justify-start">
                  <div className="max-w-[80%] rounded-lg p-3 bg-gray-100">
                    <div className="flex items-center space-x-2">
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '150ms' }}></div>
                      <div className="w-2 h-2 rounded-full bg-gray-400 animate-bounce" style={{ animationDelay: '300ms' }}></div>
                    </div>
                  </div>
                </div>
              )}
              
              {/* Ref para scroll */}
              <div ref={messagesEndRef} />
            </div>
          </div>
          
          {/* Área de entrada */}
          <div className="border-t p-4">
            {/* Prompts rápidos */}
            <div className="mb-3 flex flex-wrap gap-2">
              {quickPrompts.map(prompt => (
                <button
                  key={prompt.id}
                  className="px-3 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded-full"
                  onClick={() => handleQuickPrompt(prompt.text)}
                >
                  {prompt.text}
                </button>
              ))}
            </div>
            
            {/* Formulario de entrada */}
            <form onSubmit={handleSendMessage} className="flex space-x-2">
              <select
                className="bg-white border rounded-md px-2"
                value={selectedSymbol}
                onChange={(e) => setSelectedSymbol(e.target.value)}
              >
                <option value="">Sin símbolo</option>
                <option value="AAPL">AAPL</option>
                <option value="MSFT">MSFT</option>
                <option value="GOOGL">GOOGL</option>
                <option value="AMZN">AMZN</option>
                <option value="TSLA">TSLA</option>
              </select>
              
              <input
                type="text"
                className="flex-1 bg-white border rounded-md px-4 py-2"
                placeholder="Escribe tu mensaje..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                disabled={loading}
              />
              
              <button
                type="submit"
                className="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700 disabled:bg-indigo-400"
                disabled={loading || !inputMessage.trim()}
              >
                Enviar
              </button>
            </form>
          </div>
        </div>
      </div>
      
      {/* Panel lateral */}
      <div className="lg:col-span-1 space-y-4">
        {/* Resumen de cartera */}
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-md font-semibold mb-3">Resumen de Cartera</h3>
          
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Efectivo:</span>
              <span className="font-medium">${portfolio.cash.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-sm text-gray-500">Valor Total:</span>
              <span className="font-medium">${portfolio.totalValue.toLocaleString('es-ES', { minimumFractionDigits: 2 })}</span>
            </div>
            
            <div className="pt-2 mt-2 border-t">
              <h4 className="text-sm font-medium mb-2">Posiciones:</h4>
              
              {Object.keys(portfolio.positions).length === 0 ? (
                <p className="text-sm text-gray-500 italic">No hay posiciones abiertas</p>
              ) : (
                <div className="space-y-1 max-h-32 overflow-y-auto">
                  {Object.entries(portfolio.positions).map(([symbol, position]) => (
                    <div key={symbol} className="flex justify-between text-sm">
                      <span>{symbol}</span>
                      <span>{position.quantity} @ ${position.avgCost.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Consejos de uso */}
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-md font-semibold mb-2">Consejos para el Chat</h3>
          
          <ul className="text-sm space-y-2">
            <li>• Pregunta sobre análisis técnico de acciones</li>
            <li>• Pide recomendaciones de compra/venta</li>
            <li>• Consulta sobre indicadores de mercado</li>
            <li>• Obtén consejos de diversificación</li>
            <li>• Pregunta sobre estrategias de trading</li>
          </ul>
        </div>
        
        {/* Información de mercado */}
        <div className="bg-white rounded-lg shadow p-4">
          <h3 className="text-md font-semibold mb-2">Información de Mercado</h3>
          
          <div className="space-y-1 text-sm">
            <div className="flex justify-between">
              <span>S&P 500:</span>
              <span className="text-green-600">+0.68%</span>
            </div>
            <div className="flex justify-between">
              <span>NASDAQ:</span>
              <span className="text-green-600">+1.12%</span>
            </div>
            <div className="flex justify-between">
              <span>DOW:</span>
              <span className="text-red-600">-0.21%</span>
            </div>
            <div className="text-right text-xs text-gray-500 mt-1">
              Datos simulados para demostración
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default BrokerChat;
