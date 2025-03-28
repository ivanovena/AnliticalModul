import React, { createContext, useContext, useState, useCallback } from 'react';

// Crear contexto
const NotificationContext = createContext();

// Hook personalizado para usar el contexto
export const useNotification = () => useContext(NotificationContext);

export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([]);

  // Agregar notificación
  const addNotification = useCallback((notification) => {
    const id = Date.now().toString();
    const newNotification = {
      id,
      type: notification.type || 'info', // info, success, warning, error
      title: notification.title || '',
      message: notification.message || '',
      timestamp: new Date().toISOString(),
      read: false,
      autoClose: notification.autoClose !== false, // Por defecto las notificaciones se cierran automáticamente
    };
    
    setNotifications(prev => [newNotification, ...prev]);
    
    // Auto-cerrar después de 5 segundos si está configurado
    if (newNotification.autoClose) {
      setTimeout(() => {
        dismissNotification(id);
      }, 5000);
    }
    
    return id;
  }, []);

  // Eliminar notificación
  const dismissNotification = useCallback((id) => {
    setNotifications(prev => prev.filter(notification => notification.id !== id));
  }, []);

  // Marcar notificación como leída
  const markAsRead = useCallback((id) => {
    setNotifications(prev => 
      prev.map(notification => 
        notification.id === id ? { ...notification, read: true } : notification
      )
    );
  }, []);

  // Marcar todas las notificaciones como leídas
  const markAllAsRead = useCallback(() => {
    setNotifications(prev => 
      prev.map(notification => ({ ...notification, read: true }))
    );
  }, []);

  // Limpiar todas las notificaciones
  const clearAllNotifications = useCallback(() => {
    setNotifications([]);
  }, []);

  return (
    <NotificationContext.Provider 
      value={{
        notifications,
        addNotification,
        dismissNotification,
        markAsRead,
        markAllAsRead,
        clearAllNotifications
      }}
    >
      {children}
    </NotificationContext.Provider>
  );
};
