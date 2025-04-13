import React, { useState, useRef, useEffect } from 'react';
import {
  Box, Typography, TextField, Paper, Button,
  List, ListItem, ListItemText, CircularProgress,
  Avatar, Divider, IconButton
} from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import { chatService } from '../services/api';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

const AIChat: React.FC = () => {
  const [input, setInput] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [conversationId, setConversationId] = useState<string | undefined>(undefined);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Agregar mensaje de bienvenida inicial
  useEffect(() => {
    const welcomeMessage: Message = {
      id: 'welcome',
      text: 'Hola, soy tu asistente de trading con IA. Puedo ayudarte con análisis de mercado, recomendaciones de inversión y responder tus preguntas sobre operaciones financieras. ¿En qué puedo ayudarte hoy?',
      sender: 'bot',
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  }, []);

  // Auto-scroll al último mensaje
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value);
  };

  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    
    if (!input.trim()) return;
    
    // Agregar mensaje del usuario
    const userMessage: Message = {
      id: Date.now().toString(),
      text: input,
      sender: 'user',
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    
    try {
      // Enviar mensaje al servicio de chat
      const response = await chatService.sendMessage({
        message: input,
        conversation_id: conversationId
      });
      
      // Guardar ID de conversación para mensajes futuros
      if (response.conversation_id) {
        setConversationId(response.conversation_id);
      }
      
      // Agregar respuesta del bot
      const botMessage: Message = {
        id: Date.now().toString(),
        text: response.response,
        sender: 'bot',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending chat message:', error);
      
      // Manejar diferentes tipos de errores
      let errorText = 'Lo siento, ha ocurrido un error al procesar tu mensaje. Por favor, inténtalo de nuevo más tarde.';
      
      if (error instanceof Error) {
        if (error.message.includes('Network Error') || error.message.includes('Failed to fetch')) {
          errorText = 'No se pudo conectar con el servicio de chat. Verifica que el servidor de IA esté funcionando correctamente.';
        } else if (error.message.includes('timeout')) {
          errorText = 'La respuesta del servicio está tardando demasiado. Por favor, inténtalo de nuevo con una consulta más corta.';
        } else if (error.message.includes('500')) {
          errorText = 'Error interno en el servidor de IA. El equipo técnico ha sido notificado.';
        }
      }
      
      // Agregar mensaje de error con instrucciones más claras
      const errorMessage: Message = {
        id: Date.now().toString(),
        text: errorText + '\n\nPuedes intentar:\n1. Refrescar la página\n2. Verificar la conexión del servidor\n3. Probar con una consulta diferente',
        sender: 'bot',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">
          Asistente de Trading IA
        </Typography>
        <Button 
          size="small" 
          variant="outlined" 
          onClick={() => {
            setMessages([{
              id: 'welcome',
              text: 'Hola, soy tu asistente de trading con IA. Puedo ayudarte con análisis de mercado, recomendaciones de inversión y responder tus preguntas sobre operaciones financieras. ¿En qué puedo ayudarte hoy?',
              sender: 'bot',
              timestamp: new Date()
            }]);
            setConversationId(undefined);
          }}
        >
          Nueva Conversación
        </Button>
      </Box>
      
      {/* Mensajes */}
      <Paper 
        sx={{ 
          p: 2, 
          flexGrow: 1, 
          maxHeight: 'calc(100% - 120px)', 
          overflow: 'auto',
          mb: 2,
          bgcolor: 'background.default'
        }}
        variant="outlined"
      >
        <List>
          {messages.map((message, index) => (
            <React.Fragment key={message.id}>
              <ListItem 
                alignItems="flex-start"
                sx={{ 
                  flexDirection: message.sender === 'user' ? 'row-reverse' : 'row',
                  px: 1
                }}
              >
                <Avatar 
                  sx={{ 
                    bgcolor: message.sender === 'user' ? 'primary.main' : 'secondary.main',
                    width: 32,
                    height: 32,
                    mr: message.sender === 'user' ? 0 : 1,
                    ml: message.sender === 'user' ? 1 : 0
                  }}
                >
                  {message.sender === 'user' ? <PersonIcon /> : <SmartToyIcon />}
                </Avatar>
                
                <Paper 
                  sx={{ 
                    p: 1.5, 
                    maxWidth: '80%',
                    borderRadius: 2,
                    bgcolor: message.sender === 'user' ? 'primary.dark' : 'background.paper',
                    color: message.sender === 'user' ? 'primary.contrastText' : 'text.primary'
                  }}
                >
                  <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                    {message.text}
                  </Typography>
                  <Typography variant="caption" color="text.secondary" display="block" align="right" sx={{ mt: 0.5 }}>
                    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </Typography>
                </Paper>
              </ListItem>
              
              {index < messages.length - 1 && (
                <Box sx={{ my: 1 }} />
              )}
            </React.Fragment>
          ))}
          
          {loading && (
            <ListItem>
              <Box display="flex" alignItems="center">
                <CircularProgress size={20} sx={{ mr: 1 }} />
                <Typography variant="body2" color="text.secondary">
                  Pensando...
                </Typography>
              </Box>
            </ListItem>
          )}
          
          <div ref={messagesEndRef} />
        </List>
      </Paper>
      
      {/* Input */}
      <Box component="form" onSubmit={handleSendMessage}>
        <TextField
          fullWidth
          placeholder="Escribe tu mensaje..."
          variant="outlined"
          value={input}
          onChange={handleInputChange}
          onKeyPress={handleKeyPress}
          InputProps={{
            endAdornment: (
              <IconButton 
                color="primary" 
                onClick={() => handleSendMessage()} 
                disabled={loading || !input.trim()}
              >
                <SendIcon />
              </IconButton>
            ),
          }}
          disabled={loading}
          size="small"
        />
      </Box>
    </Box>
  );
};

export default AIChat;