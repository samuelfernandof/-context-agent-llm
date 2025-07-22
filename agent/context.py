def build_context(history):
    """
    Constrói o contexto para o agente a partir do histórico.
    """
    if not history:
        # Mensagem de sistema inicial
        return [{"role": "system", "content": "Você é um agente assistente LLM open-source, útil, didático e organizado."}]
    return history
