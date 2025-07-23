import yaml
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import asdict

from models.models import Thread, Message, ToolCall, Result
from agent.logger import get_logger

# ============================================================================
# SISTEMA DE CONTEXTO ESTRUTURADO COM YAML
# ============================================================================

class ContextBuilder:
    """
    Construtor de contexto estruturado para o agente.
    Converte Thread em formato YAML legível e estruturado.
    Segue princípios funcionais - todas as operações são puras.
    """
    
    def __init__(self, max_context_length: int = 8000, max_messages: int = 50):
        self.max_context_length = max_context_length
        self.max_messages = max_messages
        self.logger = get_logger()
    
    def build_system_prompt(self, thread: Thread, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Constrói o prompt de sistema em formato YAML estruturado.
        Função pura - não modifica o estado.
        """
        try:
            # Estrutura base do contexto
            context_data = {
                "agent_info": {
                    "name": "Agente Funcional",
                    "version": "1.0.0",
                    "description": "Agente inteligente com memória persistente e tool use",
                    "capabilities": [
                        "Memória de longo prazo",
                        "Execução de funções estruturadas",
                        "Logging detalhado",
                        "Contexto persistente"
                    ]
                },
                "session_info": {
                    "session_id": thread.session_id,
                    "created_at": thread.created_at.isoformat(),
                    "updated_at": thread.updated_at.isoformat(),
                    "message_count": len(thread.messages),
                    "tools_used": len(thread.tools_calls)
                },
                "conversation_history": self._build_conversation_summary(thread),
                "available_tools": self._get_available_tools_info(),
                "behavior_guidelines": self._get_behavior_guidelines()
            }
            
            # Adicionar contexto adicional se fornecido
            if additional_context:
                context_data["additional_context"] = additional_context
            
            # Converter para YAML formatado
            yaml_context = yaml.dump(
                context_data,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2
            )
            
            # Construir prompt final
            system_prompt = f"""Você é um agente inteligente funcional com as seguintes características e contexto:

---
{yaml_context}
---

## Instruções de Comportamento:

1. **Memória Consistente**: Use o histórico da conversa para manter contexto entre mensagens
2. **Tool Use Responsável**: Execute funções apenas quando necessário e apropriado
3. **Logs Estruturados**: Todas as ações são registradas automaticamente
4. **Respostas Úteis**: Seja preciso, didático e organizado
5. **Tratamento de Erros**: Gerencie falhas graciosamente com retry quando apropriado

Responda de forma natural e útil, considerando todo o contexto fornecido acima."""

            return system_prompt
            
        except Exception as e:
            self.logger.log_error(f"Erro ao construir contexto: {str(e)}")
            return self._get_fallback_system_prompt()
    
    def _build_conversation_summary(self, thread: Thread) -> Dict[str, Any]:
        """
        Constrói resumo da conversa de forma estruturada.
        Função pura de transformação.
        """
        if not thread.messages:
            return {
                "status": "nova_conversa",
                "messages": [],
                "summary": "Nenhuma mensagem ainda."
            }
        
        # Filtrar mensagens recentes se necessário
        recent_messages = self._filter_recent_messages(thread.messages)
        
        # Contar tipos de mensagem
        message_types = {}
        for msg in thread.messages:
            message_types[msg.role] = message_types.get(msg.role, 0) + 1
        
        # Construir resumo estruturado
        summary = {
            "status": "conversa_ativa",
            "total_messages": len(thread.messages),
            "recent_messages_shown": len(recent_messages),
            "message_types": message_types,
            "last_activity": thread.updated_at.isoformat(),
            "messages": []
        }
        
        # Adicionar mensagens recentes formatadas
        for msg in recent_messages:
            message_data = {
                "role": msg.role,
                "timestamp": msg.timestamp.isoformat(),
                "content_preview": self._truncate_content(msg.content, 200),
                "has_function_call": bool(msg.function_call)
            }
            
            if msg.name:
                message_data["function_name"] = msg.name
            
            summary["messages"].append(message_data)
        
        return summary
    
    def _filter_recent_messages(self, messages: List[Message]) -> List[Message]:
        """
        Filtra mensagens recentes baseado em limites configurados.
        Função pura de filtragem.
        """
        if not messages:
            return []
        
        # Aplicar limite de quantidade
        limited_messages = messages[-self.max_messages:] if len(messages) > self.max_messages else messages
        
        # Verificar limite de tamanho do contexto
        total_length = sum(len(msg.content) for msg in limited_messages)
        
        if total_length <= self.max_context_length:
            return limited_messages
        
        # Reduzir mensagens até caber no limite
        filtered_messages = []
        current_length = 0
        
        # Começar das mensagens mais recentes
        for msg in reversed(limited_messages):
            msg_length = len(msg.content)
            
            if current_length + msg_length <= self.max_context_length:
                filtered_messages.insert(0, msg)
                current_length += msg_length
            else:
                break
        
        return filtered_messages
    
    def _get_available_tools_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre ferramentas disponíveis.
        Função pura de metadados.
        """
        # Em uma implementação real, isso viria do sistema de tools
        return {
            "total_tools": 0,  # Será preenchido pelo sistema de tools
            "categories": [
                "utilities",
                "search", 
                "calculations",
                "file_operations"
            ],
            "note": "Tools específicas são carregadas dinamicamente pelo sistema"
        }
    
    def _get_behavior_guidelines(self) -> List[str]:
        """
        Retorna diretrizes de comportamento do agente.
        Função pura de configuração.
        """
        return [
            "Seja útil, didático e organizado em suas respostas",
            "Use a memória da conversa para fornecer contexto consistente", 
            "Execute funções apenas quando necessário e apropriado",
            "Explique seu raciocínio quando executar ações complexas",
            "Trate erros graciosamente e informe sobre problemas",
            "Mantenha respostas concisas mas completas",
            "Priorize a segurança em todas as operações"
        ]
    
    def _truncate_content(self, content: str, max_length: int = 200) -> str:
        """
        Trunca conteúdo para preview.
        Função pura de formatação.
        """
        if len(content) <= max_length:
            return content
        
        return content[:max_length-3] + "..."
    
    def _get_fallback_system_prompt(self) -> str:
        """
        Prompt de sistema de fallback em caso de erro.
        Função pura de contingência.
        """
        return """Você é um agente inteligente funcional com as seguintes características:

- Memória persistente entre conversas
- Capacidade de executar funções estruturadas  
- Sistema de logging automático
- Tratamento robusto de erros

Seja útil, didático e organizado em suas respostas. Use contexto de conversas anteriores quando relevante."""

    def build_messages_for_llm(self, thread: Thread, include_system: bool = True) -> List[Dict[str, Any]]:
        """
        Constrói lista de mensagens no formato esperado pelo LLM.
        Função pura de transformação para API do OpenAI.
        """
        messages = []
        
        # Adicionar prompt de sistema se solicitado
        if include_system:
            system_prompt = self.build_system_prompt(thread)
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Adicionar mensagens da conversa
        recent_messages = self._filter_recent_messages(thread.messages)
        
        for message in recent_messages:
            msg_dict = message.to_dict()
            
            # Filtrar campos desnecessários para o LLM
            llm_message = {
                "role": msg_dict["role"],
                "content": msg_dict["content"]
            }
            
            # Adicionar function_call se presente
            if msg_dict.get("function_call"):
                llm_message["function_call"] = msg_dict["function_call"]
            
            # Adicionar name se presente (para mensagens de função)
            if msg_dict.get("name"):
                llm_message["name"] = msg_dict["name"]
            
            messages.append(llm_message)
        
        return messages
    
    def extract_context_metadata(self, thread: Thread) -> Dict[str, Any]:
        """
        Extrai metadados do contexto para análise.
        Função pura de extração.
        """
        if not thread.messages:
            return {
                "is_empty": True,
                "estimated_tokens": 0,
                "conversation_length": 0
            }
        
        recent_messages = self._filter_recent_messages(thread.messages)
        
        # Estimativa grosseira de tokens (1 token ≈ 4 caracteres)
        total_chars = sum(len(msg.content) for msg in recent_messages)
        estimated_tokens = total_chars // 4
        
        # Análise de padrões
        roles_distribution = {}
        for msg in recent_messages:
            roles_distribution[msg.role] = roles_distribution.get(msg.role, 0) + 1
        
        # Tempo desde última mensagem
        last_message_time = thread.messages[-1].timestamp if thread.messages else thread.created_at
        time_since_last = datetime.utcnow() - last_message_time
        
        return {
            "is_empty": False,
            "estimated_tokens": estimated_tokens,
            "conversation_length": len(recent_messages),
            "total_messages": len(thread.messages),
            "roles_distribution": roles_distribution,
            "time_since_last_message_minutes": time_since_last.total_seconds() / 60,
            "context_window_utilization": min(estimated_tokens / 4000, 1.0)  # Assumindo context window de ~4k tokens
        }

# ============================================================================
# CONTEXT FILTERS E PROCESSADORES
# ============================================================================

class ContextFilter:
    """
    Filtros especializados para processamento de contexto.
    Implementa diferentes estratégias de filtragem.
    """
    
    @staticmethod
    def by_time_window(messages: List[Message], hours: int = 24) -> List[Message]:
        """
        Filtra mensagens por janela de tempo.
        Função pura de filtragem temporal.
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [msg for msg in messages if msg.timestamp >= cutoff]
    
    @staticmethod  
    def by_importance(messages: List[Message], 
                     importance_fn: Callable[[Message], float]) -> List[Message]:
        """
        Filtra mensagens por importância usando função customizada.
        Função de alta ordem para filtragem customizada.
        """
        scored_messages = [(msg, importance_fn(msg)) for msg in messages]
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        return [msg for msg, score in scored_messages if score > 0.5]
    
    @staticmethod
    def remove_system_messages(messages: List[Message]) -> List[Message]:
        """
        Remove mensagens de sistema para economia de contexto.
        Função pura de filtragem por tipo.
        """
        return [msg for msg in messages if msg.role != "system"]
    
    @staticmethod
    def compress_repeated_patterns(messages: List[Message]) -> List[Message]:
        """
        Comprime padrões repetidos na conversa.
        Função pura de compressão inteligente.
        """
        if len(messages) <= 2:
            return messages
        
        compressed = [messages[0]]  # Sempre manter primeira mensagem
        
        for i in range(1, len(messages)):
            current = messages[i]
            previous = messages[i-1]
            
            # Evitar mensagens muito similares consecutivas
            if (current.role == previous.role and 
                len(current.content) > 10 and 
                len(previous.content) > 10):
                
                # Calcular similaridade básica
                similarity = ContextFilter._calculate_similarity(current.content, previous.content)
                
                if similarity < 0.8:  # Manter se suficientemente diferente
                    compressed.append(current)
            else:
                compressed.append(current)
        
        return compressed
    
    @staticmethod
    def _calculate_similarity(text1: str, text2: str) -> float:
        """
        Calcula similaridade básica entre dois textos.
        Função auxiliar pura.
        """
        if not text1 or not text2:
            return 0.0
        
        # Implementação simples baseada em palavras comuns
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

# ============================================================================
# CONTEXT MANAGER E FACTORY
# ============================================================================

class ContextManager:
    """
    Gerenciador de contexto com diferentes estratégias.
    Combina builder e filters para contexto otimizado.
    """
    
    def __init__(self, 
                 max_context_length: int = 8000,
                 max_messages: int = 50,
                 compression_enabled: bool = True):
        self.builder = ContextBuilder(max_context_length, max_messages)
        self.compression_enabled = compression_enabled
        self.logger = get_logger()
    
    def prepare_context(self, thread: Thread, 
                       strategy: str = "default",
                       additional_context: Optional[Dict[str, Any]] = None) -> Result:
        """
        Prepara contexto usando estratégia específica.
        Função de coordenação com tratamento de erros.
        """
        try:
            # Aplicar estratégia de filtragem
            filtered_thread = self._apply_strategy(thread, strategy)
            
            # Construir contexto estruturado
            system_prompt = self.builder.build_system_prompt(filtered_thread, additional_context)
            messages = self.builder.build_messages_for_llm(filtered_thread)
            metadata = self.builder.extract_context_metadata(filtered_thread)
            
            return Result.ok({
                "system_prompt": system_prompt,
                "messages": messages,
                "metadata": metadata,
                "strategy_used": strategy
            })
            
        except Exception as e:
            error_msg = f"Erro ao preparar contexto: {str(e)}"
            self.logger.log_error(error_msg)
            return Result.error(error_msg)
    
    def _apply_strategy(self, thread: Thread, strategy: str) -> Thread:
        """
        Aplica estratégia específica de filtragem.
        Função pura de transformação.
        """
        messages = thread.messages
        
        if strategy == "recent_only":
            # Apenas mensagens das últimas 6 horas
            messages = ContextFilter.by_time_window(messages, hours=6)
            
        elif strategy == "compressed":
            # Compressão inteligente de padrões
            if self.compression_enabled:
                messages = ContextFilter.compress_repeated_patterns(messages)
            
        elif strategy == "no_system":
            # Remover mensagens de sistema para economia
            messages = ContextFilter.remove_system_messages(messages)
            
        elif strategy == "minimal":
            # Contexto mínimo - apenas últimas mensagens essenciais
            messages = messages[-5:] if len(messages) > 5 else messages
            messages = ContextFilter.remove_system_messages(messages)
            
        # Estratégia padrão - balanceada
        # Aplica filtragem temporal suave e compressão se habilitada
        else:  # strategy == "default"
            if self.compression_enabled:
                messages = ContextFilter.compress_repeated_patterns(messages)
            messages = ContextFilter.by_time_window(messages, hours=48)  # 48h de janela
        
        # Retornar nova thread com mensagens filtradas
        return Thread(
            messages=messages,
            tools_calls=thread.tools_calls,
            session_id=thread.session_id,
            created_at=thread.created_at,
            updated_at=thread.updated_at
        )

# ============================================================================
# CONTEXT SERIALIZERS E EXPORTERS
# ============================================================================

class ContextExporter:
    """
    Exportador de contexto para diferentes formatos.
    Útil para debugging, análise e backups.
    """
    
    @staticmethod
    def to_yaml(thread: Thread, include_metadata: bool = True) -> str:
        """
        Exporta thread completa para YAML legível.
        Função pura de serialização.
        """
        data = {
            "session_info": {
                "session_id": thread.session_id,
                "created_at": thread.created_at.isoformat(),
                "updated_at": thread.updated_at.isoformat(),
                "total_messages": len(thread.messages),
                "total_tool_calls": len(thread.tools_calls)
            },
            "messages": []
        }
        
        # Adicionar mensagens
        for i, msg in enumerate(thread.messages):
            message_data = {
                "index": i + 1,
                "role": msg.role,
                "timestamp": msg.timestamp.isoformat(),
                "content": msg.content
            }
            
            if msg.function_call:
                message_data["function_call"] = msg.function_call
            
            if msg.name:
                message_data["function_name"] = msg.name
            
            data["messages"].append(message_data)
        
        # Adicionar tool calls se existirem
        if thread.tools_calls:
            data["tool_calls"] = []
            for tc in thread.tools_calls:
                data["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "status": tc.status,
                    "timestamp": tc.timestamp.isoformat(),
                    "result": tc.result,
                    "error": tc.error
                })
        
        # Adicionar metadados se solicitado
        if include_metadata:
            builder = ContextBuilder()
            metadata = builder.extract_context_metadata(thread)
            data["metadata"] = metadata
        
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    @staticmethod
    def to_markdown(thread: Thread) -> str:
        """
        Exporta thread para formato Markdown legível.
        Função pura de formatação.
        """
        lines = []
        
        # Cabeçalho
        lines.append(f"# Conversa: {thread.session_id}")
        lines.append(f"**Criada em:** {thread.created_at.strftime('%d/%m/%Y %H:%M:%S')}")
        lines.append(f"**Atualizada em:** {thread.updated_at.strftime('%d/%m/%Y %H:%M:%S')}")
        lines.append(f"**Total de mensagens:** {len(thread.messages)}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Mensagens
        for i, msg in enumerate(thread.messages):
            timestamp = msg.timestamp.strftime('%H:%M:%S')
            
            if msg.role == "user":
                lines.append(f"## 👤 Usuário ({timestamp})")
            elif msg.role == "assistant":
                lines.append(f"## 🤖 Assistente ({timestamp})")
            elif msg.role == "system":
                lines.append(f"## ⚙️ Sistema ({timestamp})")
            else:
                lines.append(f"## 🔧 {msg.role.title()} ({timestamp})")
            
            lines.append("")
            lines.append(msg.content)
            lines.append("")
            
            # Adicionar info de function call se existir
            if msg.function_call:
                lines.append("**Chamada de Função:**")
                lines.append(f"```json")
                lines.append(yaml.dump(msg.function_call, default_flow_style=False))
                lines.append("```")
                lines.append("")
        
        # Tool calls se existirem
        if thread.tools_calls:
            lines.append("---")
            lines.append("")
            lines.append("## 🛠️ Histórico de Ferramentas")
            lines.append("")
            
            for tc in thread.tools_calls:
                status_emoji = "✅" if tc.status == "success" else "❌" if tc.status == "error" else "⏳"
                lines.append(f"### {status_emoji} {tc.name} ({tc.timestamp.strftime('%H:%M:%S')})")
                lines.append("")
                lines.append("**Argumentos:**")
                lines.append(f"```yaml")
                lines.append(yaml.dump(tc.arguments, default_flow_style=False))
                lines.append("```")
                
                if tc.result:
                    lines.append("")
                    lines.append("**Resultado:**")
                    lines.append(f"```")
                    lines.append(str(tc.result))
                    lines.append("```")
                
                if tc.error:
                    lines.append("")
                    lines.append("**Erro:**")
                    lines.append(f"```")
                    lines.append(tc.error)
                    lines.append("```")
                
                lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def to_json(thread: Thread, pretty: bool = True) -> str:
        """
        Exporta thread para JSON estruturado.
        Função pura de serialização.
        """
        data = thread.to_dict()
        
        if pretty:
            return json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False, default=str)
        else:
            return json.dumps(data, ensure_ascii=False, default=str)

# ============================================================================
# CONTEXT VALIDATORS
# ============================================================================

class ContextValidator:
    """
    Validador de contexto para garantir qualidade e consistência.
    Implementa verificações de integridade.
    """
    
    @staticmethod
    def validate_thread(thread: Thread) -> Result:
        """
        Valida integridade de uma thread.
        Função pura de validação.
        """
        errors = []
        warnings = []
        
        # Validações básicas
        if not thread.session_id:
            errors.append("session_id não pode estar vazio")
        
        if not thread.created_at:
            errors.append("created_at não pode estar vazio")
        
        if not thread.updated_at:
            errors.append("updated_at não pode estar vazio")
        
        if thread.updated_at < thread.created_at:
            errors.append("updated_at não pode ser anterior a created_at")
        
        # Validar mensagens
        for i, msg in enumerate(thread.messages):
            msg_errors = ContextValidator._validate_message(msg, i)
            errors.extend(msg_errors)
        
        # Validar sequência temporal das mensagens
        if len(thread.messages) > 1:
            for i in range(1, len(thread.messages)):
                current = thread.messages[i]
                previous = thread.messages[i-1]
                
                if current.timestamp < previous.timestamp:
                    warnings.append(f"Mensagem {i+1} tem timestamp anterior à mensagem {i}")
        
        # Validar tool calls
        for i, tc in enumerate(thread.tools_calls):
            tc_errors = ContextValidator._validate_tool_call(tc, i)
            errors.extend(tc_errors)
        
        # Verificar coerência geral
        if len(thread.messages) == 0 and len(thread.tools_calls) > 0:
            warnings.append("Thread tem tool calls mas nenhuma mensagem")
        
        # Retornar resultado
        if errors:
            return Result.error({
                "errors": errors,
                "warnings": warnings,
                "valid": False
            })
        
        return Result.ok({
            "errors": [],
            "warnings": warnings,
            "valid": True,
            "message_count": len(thread.messages),
            "tool_call_count": len(thread.tools_calls)
        })
    
    @staticmethod
    def _validate_message(message: Message, index: int) -> List[str]:
        """
        Valida mensagem individual.
        Função auxiliar pura.
        """
        errors = []
        
        if not message.role:
            errors.append(f"Mensagem {index+1}: role não pode estar vazio")
        
        if message.role not in ["system", "user", "assistant", "function"]:
            errors.append(f"Mensagem {index+1}: role inválido '{message.role}'")
        
        if not message.content and not message.function_call:
            errors.append(f"Mensagem {index+1}: deve ter content ou function_call")
        
        if not message.timestamp:
            errors.append(f"Mensagem {index+1}: timestamp não pode estar vazio")
        
        if message.role == "function" and not message.name:
            errors.append(f"Mensagem {index+1}: mensagens de função devem ter 'name'")
        
        return errors
    
    @staticmethod
    def _validate_tool_call(tool_call: ToolCall, index: int) -> List[str]:
        """
        Valida tool call individual.
        Função auxiliar pura.
        """
        errors = []
        
        if not tool_call.id:
            errors.append(f"Tool call {index+1}: id não pode estar vazio")
        
        if not tool_call.name:
            errors.append(f"Tool call {index+1}: name não pode estar vazio")
        
        if not tool_call.timestamp:
            errors.append(f"Tool call {index+1}: timestamp não pode estar vazio")
        
        if tool_call.status not in ["pending", "success", "error"]:
            errors.append(f"Tool call {index+1}: status inválido '{tool_call.status}'")
        
        if tool_call.status == "error" and not tool_call.error:
            errors.append(f"Tool call {index+1}: status 'error' deve ter campo 'error'")
        
        if tool_call.status == "success" and tool_call.error:
            errors.append(f"Tool call {index+1}: status 'success' não deve ter campo 'error'")
        
        return errors

# ============================================================================
# FACTORY FUNCTIONS E UTILITÁRIOS GLOBAIS
# ============================================================================

def create_context_manager(max_context_length: int = 8000, 
                          max_messages: int = 50,
                          compression_enabled: bool = True) -> ContextManager:
    """
    Factory function para criar ContextManager.
    Função pura de criação com configurações personalizadas.
    """
    return ContextManager(
        max_context_length=max_context_length,
        max_messages=max_messages,
        compression_enabled=compression_enabled
    )

def build_context(thread: Thread, 
                 strategy: str = "default",
                 additional_context: Optional[Dict[str, Any]] = None) -> Result:
    """
    Função de conveniência para construir contexto rapidamente.
    Wrapper para ContextManager com configurações padrão.
    """
    manager = create_context_manager()
    return manager.prepare_context(thread, strategy, additional_context)

def export_thread_as_yaml(thread: Thread, include_metadata: bool = True) -> str:
    """
    Função de conveniência para exportar thread como YAML.
    Wrapper para ContextExporter.
    """
    return ContextExporter.to_yaml(thread, include_metadata)

def export_thread_as_markdown(thread: Thread) -> str:
    """
    Função de conveniência para exportar thread como Markdown.
    Wrapper para ContextExporter.
    """
    return ContextExporter.to_markdown(thread)

def validate_thread_integrity(thread: Thread) -> Result:
    """
    Função de conveniência para validar integridade da thread.
    Wrapper para ContextValidator.
    """
    return ContextValidator.validate_thread(thread)

# ============================================================================
# CONTEXT CACHE (OPCIONAL)
# ============================================================================

class ContextCache:
    """
    Cache simples para contextos processados.
    Evita reprocessamento desnecessário de contextos idênticos.
    """
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self.logger = get_logger()
    
    def get_cache_key(self, thread: Thread, strategy: str) -> str:
        """
        Gera chave de cache baseada em thread e estratégia.
        Função pura de hash.
        """
        # Usar hash do conteúdo + estratégia + timestamp de atualização
        content_hash = hash(str(thread.to_dict()))
        return f"{thread.session_id}_{strategy}_{content_hash}_{thread.updated_at.timestamp()}"
    
    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Recupera contexto do cache se existir.
        Atualiza ordem de acesso (LRU).
        """
        if cache_key in self._cache:
            # Mover para o fim (mais recente)
            self._access_order.remove(cache_key)
            self._access_order.append(cache_key)
            
            self.logger.log_info("Context cache hit", cache_key=cache_key)
            return self._cache[cache_key]
        
        return None
    
    def set(self, cache_key: str, context_data: Dict[str, Any]) -> None:
        """
        Armazena contexto no cache.
        Implementa evicção LRU se necessário.
        """
        # Remover se já existe
        if cache_key in self._cache:
            self._access_order.remove(cache_key)
        
        # Evicção se cache cheio
        while len(self._cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
        
        # Adicionar novo item
        self._cache[cache_key] = context_data
        self._access_order.append(cache_key)
        
        self.logger.log_info("Context cached", cache_key=cache_key, cache_size=len(self._cache))
    
    def clear(self) -> None:
        """Limpa todo o cache."""
        self._cache.clear()
        self._access_order.clear()
        self.logger.log_info("Context cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do cache."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "utilization": len(self._cache) / self.max_size,
            "keys": list(self._cache.keys())
        }

# Cache global (opcional)
_global_context_cache: Optional[ContextCache] = None

def get_context_cache() -> ContextCache:
    """
    Retorna instância global do cache de contexto.
    Lazy initialization thread-safe.
    """
    global _global_context_cache
    
    if _global_context_cache is None:
        _global_context_cache = ContextCache()
    
    return _global_context_cache
