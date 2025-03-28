import React, { useState } from 'react';
import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { useNotification } from '../../contexts/NotificationContext';

// Importar iconos
import { 
  HomeIcon, 
  ChartBarIcon, 
  CubeIcon, 
  LightBulbIcon, 
  ChatBubbleLeftRightIcon, 
  GlobeAltIcon, 
  BellIcon 
} from '@heroicons/react/24/outline';

const MainLayout = () => {
  const location = useLocation();
  const { notifications } = useNotification();
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Clase activa para NavLink
  const activeClass = "bg-indigo-800 text-white";
  const inactiveClass = "text-indigo-100 hover:bg-indigo-700";

  // Navegación principal
  const navigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Simulador Trading', href: '/trading', icon: ChartBarIcon },
    { name: 'Estado de Modelos', href: '/models', icon: CubeIcon },
    { name: 'Análisis de Predicciones', href: '/predictions', icon: LightBulbIcon },
    { name: 'Chat con Broker IA', href: '/broker-chat', icon: ChatBubbleLeftRightIcon },
    { name: 'Análisis de Mercado', href: '/market', icon: GlobeAltIcon },
  ];

  return (
    <div className="h-screen flex overflow-hidden bg-gray-100">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'block' : 'hidden'} md:block md:flex-shrink-0`}>
        <div className="h-full flex flex-col w-64 bg-indigo-900">
          {/* Logo */}
          <div className="flex items-center h-16 px-4 bg-indigo-950">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-white rounded-md flex items-center justify-center">
                <ChartBarIcon className="w-5 h-5 text-indigo-600" />
              </div>
              <span className="text-white font-semibold text-lg">Trading Platform</span>
            </div>
          </div>

          {/* Enlaces de navegación */}
          <nav className="flex-1 px-2 py-4 space-y-1 overflow-y-auto">
            {navigation.map((item) => (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) => 
                  `flex items-center px-2 py-2 text-sm font-medium rounded-md ${
                    isActive ? activeClass : inactiveClass
                  }`
                }
              >
                <item.icon className="mr-3 flex-shrink-0 h-6 w-6" />
                {item.name}
              </NavLink>
            ))}
          </nav>

          {/* Footer del sidebar */}
          <div className="px-2 py-4 bg-indigo-950">
            <div className="flex items-center px-2 py-2 text-sm font-medium text-indigo-100">
              <div className="w-2 h-2 rounded-full bg-green-400 mr-2"></div>
              <span>Sistema en línea</span>
            </div>
          </div>
        </div>
      </div>

      {/* Contenido principal */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* Barra superior */}
        <header className="w-full">
          <div className="relative z-10 flex-shrink-0 h-16 bg-white shadow-sm flex">
            {/* Botón de toggle sidebar */}
            <button
              className="px-4 text-gray-500 focus:outline-none md:hidden"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>

            {/* Título de la página actual */}
            <div className="flex-1 flex items-center justify-between px-4">
              <h1 className="text-xl font-semibold text-gray-900">
                {navigation.find(item => item.href === location.pathname)?.name || 'Dashboard'}
              </h1>

              {/* Notificaciones */}
              <div className="ml-4 flex items-center md:ml-6">
                <button className="p-1 rounded-full text-gray-400 hover:text-gray-500 focus:outline-none">
                  <BellIcon className="h-6 w-6" />
                  {notifications.length > 0 && (
                    <span className="absolute top-0 right-0 -mt-1 -mr-1 flex h-4 w-4 items-center justify-center rounded-full bg-red-500 text-xs text-white">
                      {notifications.length}
                    </span>
                  )}
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Área de contenido principal */}
        <main className="flex-1 overflow-y-auto bg-gray-100 p-4">
          <Outlet />
        </main>
      </div>
    </div>
  );
};

export default MainLayout;
