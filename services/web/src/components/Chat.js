import React, { useState } from 'react';
import axios from 'axios';

function Chat() {
  const [message, setMessage] = useState("");
  const [chatLog, setChatLog] = useState([]);

  const sendMessage = async () => {
    try {
      const response = await axios.post('http://localhost:8000/chat', { message });
      setChatLog([...chatLog, { message, response: response.data.response }]);
      setMessage("");
    } catch (error) {
      console.error("Error sending message", error);
    }
  };

  return (
    <div className="chat">
      <h2>Chat con el Broker</h2>
      <div className="chat-log">
        {chatLog.map((entry, index) => (
          <div key={index}>
            <strong>Usuario:</strong> {entry.message}<br/>
            <strong>Broker:</strong> {entry.response}
          </div>
        ))}
      </div>
      <input
        type="text"
        placeholder="Escribe tu mensaje..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      />
      <button onClick={sendMessage}>Enviar</button>
    </div>
  );
}

export default Chat;
