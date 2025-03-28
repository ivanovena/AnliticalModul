// src/utils/formatters.js

/**
 * Formatea un número con separadores de miles
 * @param {number} number - El número a formatear
 * @param {number} decimals - Número de decimales a mostrar
 * @return {string} El número formateado
 */
export const formatNumberWithCommas = (number, decimals = 2) => {
  if (number === null || number === undefined) return '-';
  
  try {
    return number.toLocaleString('es-ES', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals
    });
  } catch (error) {
    console.error('Error formatting number:', error);
    return number.toString();
  }
};

/**
 * Formatea un número como precio en dólares
 * @param {number} price - El precio a formatear
 * @return {string} El precio formateado
 */
export const formatPrice = (price) => {
  if (price === null || price === undefined) return '$-';
  
  try {
    return `$${formatNumberWithCommas(price)}`;
  } catch (error) {
    console.error('Error formatting price:', error);
    return `$${price}`;
  }
};

/**
 * Formatea un porcentaje con signo y color
 * @param {number} percentage - El porcentaje a formatear
 * @param {boolean} includeSign - Si debe incluir el signo + para valores positivos
 * @return {object} Objeto con valor y clase CSS
 */
export const formatPercentage = (percentage, includeSign = true) => {
  if (percentage === null || percentage === undefined) {
    return { value: '-', className: '' };
  }
  
  try {
    let formattedValue = percentage.toFixed(2) + '%';
    let className = '';
    
    if (percentage > 0) {
      className = 'positive-value';
      if (includeSign) formattedValue = '+' + formattedValue;
    } else if (percentage < 0) {
      className = 'negative-value';
    }
    
    return { value: formattedValue, className };
  } catch (error) {
    console.error('Error formatting percentage:', error);
    return { value: percentage.toString() + '%', className: '' };
  }
};

/**
 * Formatea una fecha en formato local
 * @param {string|Date} date - La fecha a formatear
 * @param {string} format - El formato deseado (corto, largo, hora)
 * @return {string} La fecha formateada
 */
export const formatDate = (date, format = 'corto') => {
  if (!date) return '-';
  
  try {
    const dateObj = new Date(date);
    
    switch (format) {
      case 'corto':
        return dateObj.toLocaleDateString('es-ES');
      case 'largo':
        return dateObj.toLocaleDateString('es-ES', {
          year: 'numeric',
          month: 'long',
          day: 'numeric'
        });
      case 'hora':
        return dateObj.toLocaleString('es-ES');
      default:
        return dateObj.toLocaleDateString('es-ES');
    }
  } catch (error) {
    console.error('Error formatting date:', error);
    return date.toString();
  }
};