def _generate_response(self, message: str, conversation_id: str) -> str:
    """
    Genera una respuesta utilizando el modelo Llama o reglas predefinidas
    
    Args:
        message: Mensaje del usuario
        conversation_id: ID de la conversación
        
    Returns:
        Respuesta generada
    """
    # Si el mensaje es vacío o muy corto
    if not message or len(message) < 2:
        return "Por favor, envía un mensaje más detallado para poder ayudarte."
    
    # Detectar intenciones específicas
    intent, params = self._detect_intent(message)
    
    # Procesar según la intención detectada
    if intent == "get_strategy":
        return self._get_investment_strategy(params.get("symbol"))
    elif intent == "get_portfolio":
        return self._get_portfolio_info()
    elif intent == "get_cash":
        return self._get_cash_info()
    elif intent == "place_order":
        return self._handle_order_intent(params)
    elif intent == "get_risk":
        return self._get_risk_metrics()
    elif intent == "get_market_data":
        return self._get_market_data(params.get("symbol"))
    elif intent == "get_predictions":
        return self._get_predictions(params.get("symbol"))
    elif intent == "get_help":
        return self._get_help_info()
    
    # Si llegamos aquí, intentar generar respuesta con Llama
    if self.model:
        try:
            # Obtener contexto de la conversación
            conversation = self.conversations[conversation_id]
            prompt = self._build_llama_prompt(conversation, message)
            
            # Generar respuesta con el modelo
            response = self.model(
                prompt,
                max_tokens=512,
                stop=["<|user|>", "<|endoftext|>", "<|system|>"],
                temperature=0.7,
                repeat_penalty=1.1
            )
            
            # Extraer texto generado
            generated_text = response["choices"][0]["text"].strip()
            
            # Si la respuesta está vacía o es muy corta, usar respuesta por defecto
            if not generated_text or len(generated_text) < 5:
                return "Entiendo tu consulta. ¿Podrías proporcionar más detalles para poder ayudarte mejor?"
            
            return generated_text
        except Exception as e:
            logger.error(f"Error generando respuesta con modelo: {e}")
            return self._get_fallback_response(message)
    
    # Respuesta por defecto si el modelo no está disponible
    return self._get_fallback_response(message)