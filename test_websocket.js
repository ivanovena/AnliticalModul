// Script para probar la conexión WebSocket al servicio de streaming

// URL del servicio WebSocket
const wsUrl = "ws://localhost:8090";

// Crear una nueva conexión WebSocket
console.log(`Intentando conectar a: ${wsUrl}`);
const socket = new WebSocket(wsUrl);

// Manejador para cuando se abre la conexión
socket.onopen = function(event) {
  console.log("Conexión establecida con éxito!");
  
  // Enviar un mensaje de prueba (ping)
  const pingMessage = {
    action: "ping",
    timestamp: new Date().toISOString()
  };
  
  console.log("Enviando mensaje ping:", pingMessage);
  socket.send(JSON.stringify(pingMessage));
  
  // También suscribirse a un símbolo
  setTimeout(() => {
    const subscribeMessage = {
      action: "subscribe",
      topic: "prediction:AAPL",
      timestamp: new Date().toISOString()
    };
    
    console.log("Enviando suscripción:", subscribeMessage);
    socket.send(JSON.stringify(subscribeMessage));
  }, 1000);
};

// Manejador para cuando se reciben mensajes
socket.onmessage = function(event) {
  console.log("Mensaje recibido:");
  try {
    // Intentar parsear como JSON
    const data = JSON.parse(event.data);
    console.log(JSON.stringify(data, null, 2));
  } catch (e) {
    // Si no es JSON, mostrar como texto
    console.log(event.data);
  }
};

// Manejador para errores
socket.onerror = function(error) {
  console.error("Error en la conexión WebSocket:", error);
};

// Manejador para cuando se cierra la conexión
socket.onclose = function(event) {
  console.log(`Conexión cerrada. Código: ${event.code}, Razón: ${event.reason || "No especificada"}`);
};

// Mantener el script ejecutándose
console.log("Script iniciado. Presiona Ctrl+C para terminar."); 