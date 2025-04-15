const WebSocket = require('ws');

/**
 * Script de diagnóstico mejorado para probar conexiones WebSocket
 * Este script intenta conectarse a los diferentes endpoints WebSocket
 * del sistema y reporta problemas de conexión o respuesta.
 */

// Configuración de prueba
const CONFIG = {
  // Endpoints WebSocket a probar
  endpoints: [
    {
      name: 'Broker (Market Data)',
      url: 'ws://localhost:8100',
      subscriptionTopic: 'market',
      expectedMessageTypes: ['welcome', 'subscription', 'market_data']
    },
    {
      name: 'Streaming (Predicciones)',
      url: 'ws://localhost:8090',
      subscriptionTopic: 'prediction',
      expectedMessageTypes: ['welcome', 'subscription', 'prediction']
    }
  ],
  // Configuración de tiempos
  timeouts: {
    connection: 8000,     // 8 segundos para establecer conexión
    message: 10000,       // 10 segundos para recibir respuesta
    testDuration: 15000   // 15 segundos de duración total por prueba
  },
  // Headers para la conexión WebSocket
  headers: {
    'Origin': 'http://localhost:3000',
    'User-Agent': 'Mozilla/5.0 (WebSocket Test Script)',
  }
};

// Colores para la consola
const colors = {
  reset: "\x1b[0m",
  bright: "\x1b[1m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
};

// Funciones auxiliares
function log(message, color = colors.reset) {
  console.log(`${color}${message}${colors.reset}`);
}

function logError(message) {
  log(`❌ ERROR: ${message}`, colors.red);
}

function logSuccess(message) {
  log(`✅ ${message}`, colors.green);
}

function logWarning(message) {
  log(`⚠️ ${message}`, colors.yellow);
}

function logInfo(message) {
  log(`ℹ️ ${message}`, colors.cyan);
}

// Función para probar un endpoint WebSocket
async function testWebSocketEndpoint(endpoint) {
  return new Promise((resolve) => {
    log(`\n${colors.bright}=== Probando conexión a ${endpoint.name}: ${endpoint.url} ===${colors.reset}`);
    
    // Resultados de la prueba
    const results = {
      name: endpoint.name,
      url: endpoint.url,
      connected: false,
      errors: [],
      warnings: [],
      receivedMessages: [],
      receivedTypes: new Set(),
      subscription: false,
      connectionTime: null,
      firstMessageTime: null,
      passedTests: 0,
      totalTests: 3 // Conexión, suscripción, recepción de mensaje
    };
    
    let firstMessageReceived = false;
    let connectionStart = Date.now();
    
    try {
      // Crear WebSocket con timeout
      const connectionTimeout = setTimeout(() => {
        if (!results.connected) {
          results.errors.push('Timeout al intentar conectar');
          logError(`Timeout al conectar a ${endpoint.url} (${CONFIG.timeouts.connection}ms)`);
          if (ws && ws.readyState !== WebSocket.CLOSED) {
            ws.close();
          }
        }
      }, CONFIG.timeouts.connection);
      
      // Crear WebSocket
      logInfo(`Conectando a ${endpoint.url}...`);
      const ws = new WebSocket(endpoint.url, {
        headers: CONFIG.headers,
        timeout: CONFIG.timeouts.connection
      });
      
      // Configurar evento de apertura
      ws.on('open', () => {
        clearTimeout(connectionTimeout);
        results.connected = true;
        results.connectionTime = Date.now() - connectionStart;
        results.passedTests++;
        
        logSuccess(`Conectado a ${endpoint.url} (${results.connectionTime}ms)`);
        
        // Enviar suscripción
        const subscriptionMessage = JSON.stringify({
          action: 'subscribe',
          topic: endpoint.subscriptionTopic,
          timestamp: new Date().toISOString()
        });
        
        logInfo(`Enviando mensaje de suscripción: ${subscriptionMessage}`);
        ws.send(subscriptionMessage);
        
        // Configurar timeout para recibir respuesta
        setTimeout(() => {
          if (!firstMessageReceived) {
            results.warnings.push('No se recibió ningún mensaje en el tiempo esperado');
            logWarning(`No se recibió respuesta después de ${CONFIG.timeouts.message}ms`);
          }
        }, CONFIG.timeouts.message);
      });
      
      // Configurar evento de mensaje
      ws.on('message', (data) => {
        try {
          // Marcar tiempo del primer mensaje
          if (!firstMessageReceived) {
            firstMessageReceived = true;
            results.firstMessageTime = Date.now() - connectionStart;
            logSuccess(`Primer mensaje recibido en ${results.firstMessageTime}ms`);
          }
          
          // Intentar parsear JSON
          let message;
          try {
            message = JSON.parse(data.toString());
            logInfo(`Mensaje recibido: ${JSON.stringify(message, null, 2)}`);
          } catch (e) {
            message = { raw: data.toString() };
            logWarning(`Mensaje recibido no es JSON válido: ${data.toString()}`);
          }
          
          // Almacenar mensaje y tipo
          results.receivedMessages.push(message);
          
          // Identificar tipo de mensaje
          const messageType = message.type || 'unknown';
          results.receivedTypes.add(messageType);
          
          // Verificar si es una confirmación de suscripción
          if (messageType === 'subscription') {
            results.subscription = true;
            results.passedTests++;
            logSuccess(`Suscripción confirmada para ${endpoint.subscriptionTopic}`);
          }
          
          // Comprobar si hemos recibido un mensaje de datos (no solo bienvenida o suscripción)
          const dataMessageTypes = endpoint.expectedMessageTypes.filter(t => 
            t !== 'welcome' && t !== 'subscription');
          
          if (dataMessageTypes.includes(messageType)) {
            results.passedTests++;
            logSuccess(`Mensaje de datos (${messageType}) recibido correctamente`);
          }
        } catch (e) {
          results.errors.push(`Error procesando mensaje: ${e.message}`);
          logError(`Error procesando mensaje: ${e.message}`);
        }
      });
      
      // Configurar evento de error
      ws.on('error', (error) => {
        results.errors.push(`Error de WebSocket: ${error.message || 'Error desconocido'}`);
        logError(`Error de conexión: ${error.message || 'Error desconocido'}`);
      });
      
      // Configurar evento de cierre
      ws.on('close', (code, reason) => {
        logInfo(`Conexión cerrada: código ${code}, razón: ${reason || 'sin razón'}`);
        if (code !== 1000 && code !== 1001) {
          results.warnings.push(`Conexión cerrada con código ${code}: ${reason || 'sin razón'}`);
        }
      });
      
      // Finalizar prueba después del tiempo configurado
      setTimeout(() => {
        try {
          if (ws.readyState === WebSocket.OPEN) {
            ws.close();
          }
          
          // Evaluación final
          const receivedTypesArray = Array.from(results.receivedTypes);
          
          if (!results.connected) {
            logError(`No se pudo conectar a ${endpoint.url}`);
          } else if (!results.subscription) {
            logWarning(`Conectado pero no se confirmó la suscripción a ${endpoint.subscriptionTopic}`);
          } else if (results.receivedMessages.length === 0) {
            logWarning(`Conectado y suscrito, pero no se recibieron mensajes`);
          } else {
            logSuccess(`Prueba completada: conexión establecida, suscripción confirmada, ${results.receivedMessages.length} mensajes recibidos`);
            logInfo(`Tipos de mensajes recibidos: ${receivedTypesArray.join(', ')}`);
          }
          
          // Estado final
          log(`\n${colors.bright}=== Resultado para ${endpoint.name} ===${colors.reset}`);
          log(`Tests pasados: ${results.passedTests}/${results.totalTests}`, 
              results.passedTests === results.totalTests ? colors.green : colors.yellow);
          
          if (results.errors.length > 0) {
            logError(`Errores (${results.errors.length}):`);
            results.errors.forEach(err => logError(`  - ${err}`));
          }
          
          if (results.warnings.length > 0) {
            logWarning(`Advertencias (${results.warnings.length}):`);
            results.warnings.forEach(warn => logWarning(`  - ${warn}`));
          }
          
          // Recomendaciones
          if (results.errors.length > 0 || results.warnings.length > 0) {
            log(`\n${colors.bright}Recomendaciones:${colors.reset}`);
            
            if (!results.connected) {
              log("- Verifica que el servicio esté corriendo y el puerto esté correctamente expuesto en Docker", colors.cyan);
              log("- Confirma que no hay firewall o proxy bloqueando las conexiones WebSocket", colors.cyan);
              log("- Revisa los logs del servicio para errores relacionados con el WebSocket", colors.cyan);
            } else if (!results.subscription) {
              log("- Verifica el formato del mensaje de suscripción", colors.cyan);
              log("- Confirma que el servicio está procesando correctamente las suscripciones", colors.cyan);
              log("- Revisa los logs del servidor para ver si hay errores al procesar la suscripción", colors.cyan);
            } else if (results.receivedMessages.length === 0) {
              log("- Verifica que el servicio está generando datos para enviar", colors.cyan);
              log("- Aumenta el tiempo de espera para mensajes si el servicio es lento", colors.cyan);
            }
          } else {
            logSuccess("✨ Todo funciona correctamente! No se necesitan ajustes.");
          }
          
          // Enviar resultados
          resolve(results);
        } catch (finalError) {
          logError(`Error durante la finalización: ${finalError.message}`);
          results.errors.push(`Error durante finalización: ${finalError.message}`);
          resolve(results);
        }
      }, CONFIG.timeouts.testDuration);
      
    } catch (e) {
      logError(`Error al crear WebSocket: ${e.message}`);
      results.errors.push(`Error al crear WebSocket: ${e.message}`);
      resolve(results);
    }
  });
}

// Función principal
async function runTests() {
  log(`${colors.bright}Iniciando pruebas de conexión WebSocket...${colors.reset}`);
  log(`Fecha y hora: ${new Date().toISOString()}`);
  log(`Sistema operativo: ${process.platform} ${process.arch}`);
  log(`Node.js: ${process.version}`);
  
  const results = [];
  
  for (const endpoint of CONFIG.endpoints) {
    try {
      const result = await testWebSocketEndpoint(endpoint);
      results.push(result);
    } catch (e) {
      logError(`Error general en prueba de ${endpoint.name}: ${e.message}`);
    }
  }
  
  // Resumen final
  log(`\n${colors.bright}===========================================${colors.reset}`);
  log(`${colors.bright}Resumen de las pruebas WebSocket${colors.reset}`);
  log(`${colors.bright}===========================================${colors.reset}`);
  
  let allPassed = true;
  
  results.forEach(result => {
    const status = result.errors.length === 0 && result.passedTests === result.totalTests 
      ? `${colors.green}ÉXITO${colors.reset}` 
      : `${colors.red}FALLÓ${colors.reset}`;
    
    log(`${result.name}: ${status} - ${result.passedTests}/${result.totalTests} tests pasados`);
    allPassed = allPassed && (result.errors.length === 0 && result.passedTests === result.totalTests);
  });
  
  if (allPassed) {
    logSuccess("\n✅ Todas las pruebas pasaron correctamente!");
  } else {
    logWarning("\n⚠️ Una o más pruebas fallaron. Revisa los detalles anteriores para resolver los problemas.");
  }
}

// Ejecutar pruebas
runTests().catch(e => {
  logError(`Error general: ${e.message}`);
}); 