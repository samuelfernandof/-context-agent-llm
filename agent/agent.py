import os
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Generator
from dataclasses import dataclass, field
import openai
from dotenv import load_dotenv

from models.models import Thread, Message, ToolCall, Event, Result, MessageRole
from agent.memory import get_memory, AgentMemory
from agent.context import create_context_manager, ContextManager
from agent.tools import get_tool_registry, ToolRegistry, parse_function_call
from agent.logger import get_logger, log_operation, StructuredLogger

# Carregar variáveis de ambiente
load_dotenv()

# ============================================================================
# CONFIGURAÇÃO E TIPOS
# ============================================================================

@dataclass(frozen=True)
class AgentConfig:
    """
    Configuração imutável do agente.
    Centraliza todos os parâmetros configuráveis.
    """
    # LLM Configuration
    model: str = "mistralai/mistral-7b-instruct"
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    api_base: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 2000
    temperature: float = 0.7
    
    # Agent Behavior
    max_conversation_turns: int = 100
    max_function_calls_per_turn: int = 5
    enable_function_calling: bool = True
    auto_save_memory: bool = True
    
    # Context Management
    max_context_length: int = 8000
    max_messages_in_context: int = 50
    context_strategy: str = "default"  # "default", "recent_only", "compressed", "minimal"
    
    # Retry and Resilience
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True
    
    # Memory and Persistence
    memory_db_path: str = "memory.db"
    enable_memory_persistence: bool = True
    auto_cleanup_old_data: bool = False
    cleanup_days_threshold: int = 30
    
    def validate(self) -> Result:
        """Valida configuração do agente"""
        errors = []
        
        if not self.api_key:
            errors.append("OPENROUTER_API_KEY não configurada")
        
        if self.max_tokens < 100:
            errors.append("max_tokens deve ser pelo menos 100")
        
        if self.temperature < 0 or self.temperature > 2:
            errors.append("temperature deve estar entre 0 e 2")
        
        if self.max_conversation_turns < 1:
            errors.append("max_conversation_turns deve ser pelo menos 1")
        
        if errors:
            return Result.error("; ".join(errors))
        
        return Result.ok("Configuração válida")

# ============================================================================
# CLASSE PRINCIPAL DO AGENTE
# ============================================================================

class FunctionalAgent:
    """
    Agente inteligente funcional com memória persistente e tool use.
    
    Implementa:
    - Loop autônomo com retry/fallback
    - Memória de longo prazo
    - Execução estruturada de ferramentas
    - Logging detalhado
    - Gerenciamento de contexto inteligente
    - Tratamento robusto de erros
    """
    
    def __init__(self, config: Optional[AgentConfig] = None, session_id: Optional[str] = None):
        # Configuração
        self.config = config or AgentConfig()
        
        # Validar configuração
        config_validation = self.config.validate()
        if not config_validation.success:
            raise ValueError(f"Configuração inválida: {config_validation.error}")
        
        # Componentes principais
        self.logger = get_logger()
        self.memory = get_memory(self.config.memory_db_path)
        self.context_manager = create_context_manager(
            max_context_length=self.config.max_context_length,
            max_messages=self.config.max_messages_in_context
        )
        self.tool_registry = get_tool_registry()
        
        # Estado do agente
        self.session_id = session_id or f"session_{datetime.utcnow().timestamp()}"
        self.current_thread: Optional[Thread] = None
        self.is_running = False
        self.total_turns = 0
        
        # Configurar OpenAI
        openai.api_key = self.config.api_key
        openai.api_base = self.config.api_base
        
        # Inicialização
        self._initialize_agent()
    
    def _initialize_agent(self) -> None:
        """
        Inicializa o agente e carrega memória existente.
        Efeito colateral controlado de inicialização.
        """
        with log_operation(self.logger, "initialize_agent", self.session_id):
            try:
                # Tentar carregar thread existente
                if self.config.enable_memory_persistence:
                    memory_result = self.memory.load_latest_thread()
                    
                    if memory_result.success:
                        self.current_thread = memory_result.data
                        self.session_id = self.current_thread.session_id
                        self.logger.log_info(
                            f"Thread carregada da memória: {self.session_id}",
                            message_count=len(self.current_thread.messages)
                        )
                    else:
                        # Criar nova thread
                        self.current_thread = Thread(session_id=self.session_id)
                        self.logger.log_info(f"Nova thread criada: {self.session_id}")
                else:
                    # Memória desabilitada - sempre nova thread
                    self.current_thread = Thread(session_id=self.session_id)
                
                # Log de inicialização
                self.logger.log_info(
                    "Agente inicializado com sucesso",
                    session_id=self.session_id,
                    model=self.config.model,
                    tools_available=len(self.tool_registry.get_available_tools()),
                    memory_persistence=self.config.enable_memory_persistence
                )
                
            except Exception as e:
                error_msg = f"Erro na inicialização do agente: {str(e)}"
                self.logger.log_error(error_msg)
                raise RuntimeError(error_msg)
    
    def start_conversation(self) -> None:
        """
        Inicia loop de conversa interativo.
        Função principal para uso em modo CLI.
        """
        self.is_running = True
        self.logger.log_info("Iniciando conversa interativa", session_id=self.session_id)
        
        print(f"🤖 Agente Funcional v1.0 iniciado!")
        print(f"📍 Sessão: {self.session_id}")
        print(f"🧠 Modelo: {self.config.model}")
        print(f"🛠️  Ferramentas disponíveis: {len(self.tool_registry.get_available_tools())}")
        print("Digite 'quit', 'exit' ou 'sair' para encerrar.\n")
        
        try:
            while self.is_running and self.total_turns < self.config.max_conversation_turns:
                try:
                    # Input do usuário
                    user_input = input("👤 Você: ").strip()
                    
                    # Comandos especiais
                    if user_input.lower() in ['quit', 'exit', 'sair', 'q']:
                        break
                    
                    if user_input.lower() in ['help', 'ajuda', 'h']:
                        self._show_help()
                        continue
                    
                    if user_input.lower().startswith('/'):
                        self._handle_command(user_input)
                        continue
                    
                    if not user_input:
                        print("💭 Digite uma mensagem ou 'help' para ver comandos disponíveis.\n")
                        continue
                    
                    # Processar mensagem do usuário
                    response_result = self.process_user_message(user_input)
                    
                    if response_result.success:
                        print(f"🤖 Agente: {response_result.data}")
                    else:
                        print(f"❌ Erro: {response_result.error}")
                    
                    print()  # Linha em branco para separação
                    
                except KeyboardInterrupt:
                    print("\n🛑 Interrompido pelo usuário.")
                    break
                except EOFError:
                    print("\n👋 Entrada finalizada.")
                    break
                except Exception as e:
                    self.logger.log_error(f"Erro no loop principal: {str(e)}")
                    print(f"❌ Erro inesperado: {str(e)}")
                    continue
        
        finally:
            self._shutdown()
    
    def process_user_message(self, message: str) -> Result:
        """
        Processa uma mensagem do usuário e retorna resposta.
        Função principal para integração programática.
        """
        if not self.current_thread:
            return Result.error("Agente não inicializado")
        
        start_time = datetime.utcnow()
        
        with log_operation(self.logger, "process_user_message", self.session_id):
            try:
                # Adicionar mensagem do usuário à thread
                user_message = Message(
                    role="user",
                    content=message,
                    timestamp=start_time
                )
                
                self.current_thread = self.current_thread.add_message(user_message)
                self.total_turns += 1
                
                # Log da mensagem do usuário
                self.logger.log_event(Event(
                    type="user_message",
                    data={"content": message, "turn": self.total_turns},
                    session_id=self.session_id
                ))
                
                # Gerar resposta do agente
                response_result = self._generate_agent_response()
                
                if not response_result.success:
                    return response_result
                
                assistant_response = response_result.data
                
                # Adicionar resposta à thread
                assistant_message = Message(
                    role="assistant",
                    content=assistant_response,
                    timestamp=datetime.utcnow()
                )
                
                self.current_thread = self.current_thread.add_message(assistant_message)
                
                # Salvar na memória se habilitado
                if self.config.auto_save_memory and self.config.enable_memory_persistence:
                    self._save_to_memory()
                
                # Log da resposta
                self.logger.log_event(Event(
                    type="assistant_response",
                    data={"content": assistant_response, "turn": self.total_turns},
                    session_id=self.session_id
                ))
                
                return Result.ok(assistant_response)
                
            except Exception as e:
                error_msg = f"Erro ao processar mensagem: {str(e)}"
                self.logger.log_error(error_msg, session_id=self.session_id)
                return Result.error(error_msg)
    
    def _generate_agent_response(self) -> Result:
        """
        Gera resposta do agente usando o LLM com tool use.
        Implementa retry logic e tratamento de function calls.
        """
        for attempt in range(self.config.max_retries):
            try:
                # Preparar contexto
                context_result = self.context_manager.prepare_context(
                    self.current_thread,
                    strategy=self.config.context_strategy
                )
                
                if not context_result.success:
                    return Result.error(f"Erro ao preparar contexto: {context_result.error}")
                
                context_data = context_result.data
                messages = context_data["messages"]
                
                # Preparar ferramentas se habilitadas
                tools = None
                if self.config.enable_function_calling:
                    available_tools = self.tool_registry.get_available_tools()
                    if available_tools:
                        tools = available_tools
                
                # Chamar LLM
                llm_result = self._call_llm(messages, tools)
                
                if not llm_result.success:
                    if attempt < self.config.max_retries - 1:
                        delay = self._calculate_retry_delay(attempt)
                        self.logger.log_info(f"Tentativa {attempt + 1} falhou, tentando novamente em {delay}s")
                        time.sleep(delay)
                        continue
                    else:
                        return llm_result
                
                response_data = llm_result.data
                
                # Processar resposta (pode incluir function calls)
                final_response = self._process_llm_response(response_data)
                
                return Result.ok(final_response)
                
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    delay = self._calculate_retry_delay(attempt)
                    self.logger.log_error(f"Tentativa {attempt + 1} falhou: {str(e)}, tentando novamente em {delay}s")
                    time.sleep(delay)
                    continue
                else:
                    return Result.error(f"Falha após {self.config.max_retries} tentativas: {str(e)}")
        
        return Result.error("Número máximo de tentativas excedido")
    
    def _call_llm(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Result:
        """
        Chama o LLM (OpenAI/OpenRouter) com os parâmetros configurados.
        Função com efeito colateral de rede.
        """
        try:
            # Preparar parâmetros da chamada
            call_params = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            
            # Adicionar ferramentas se disponíveis
            if tools and self.config.enable_function_calling:
                call_params["functions"] = tools
                call_params["function_call"] = "auto"
            
            # Log da chamada (sem exibir API key)
            self.logger.log_info(
                "Chamando LLM",
                model=self.config.model,
                message_count=len(messages),
                tools_count=len(tools) if tools else 0,
                session_id=self.session_id
            )
            
            # Fazer a chamada
            response = openai.ChatCompletion.create(**call_params)
            
            # Extrair dados da resposta
            choice = response.choices[0]
            message = choice.message
            
            response_data = {
                "content": message.get("content", ""),
                "function_call": message.get("function_call"),
                "finish_reason": choice.finish_reason,
                "usage": response.get("usage", {})
            }
            
            # Log do uso de tokens
            usage = response_data["usage"]
            if usage:
                self.logger.log_info(
                    "Resposta LLM recebida",
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    finish_reason=response_data["finish_reason"],
                    session_id=self.session_id
                )
            
            return Result.ok(response_data)
            
        except Exception as e:
            error_msg = f"Erro na chamada LLM: {str(e)}"
            self.logger.log_error(error_msg, session_id=self.session_id)
            return Result.error(error_msg)
    
    def _process_llm_response(self, response_data: Dict[str, Any]) -> str:
        """
        Processa resposta do LLM, incluindo execução de function calls.
        Retorna resposta final formatada.
        """
        content = response_data.get("content", "")
        function_call = response_data.get("function_call")
        
        # Se não há function call, retornar conteúdo diretamente
        if not function_call:
            return content or "Desculpe, não consegui gerar uma resposta."
        
        try:
            # Processar function call
            function_result = self._execute_function_call(function_call)
            
            if function_result.success:
                # Adicionar resultado da função à thread
                function_message = Message(
                    role="function",
                    name=function_call["name"],
                    content=str(function_result.data),
                    timestamp=datetime.utcnow()
                )
                
                self.current_thread = self.current_thread.add_message(function_message)
                
                # Se o LLM também retornou conteúdo junto com function call
                if content:
                    return f"{content}\n\n[Resultado da função {function_call['name']}: {function_result.data}]"
                else:
                    # Gerar nova resposta considerando o resultado da função
                    return self._generate_response_with_function_result(function_call["name"], function_result.data)
            
            else:
                error_msg = f"Erro ao executar função {function_call['name']}: {function_result.error}"
                self.logger.log_error(error_msg, session_id=self.session_id)
                return f"Desculpe, ocorreu um erro ao executar a função solicitada: {function_result.error}"
                
        except Exception as e:
            error_msg = f"Erro no processamento de function call: {str(e)}"
            self.logger.log_error(error_msg, session_id=self.session_id)
            return f"Desculpe, ocorreu um erro inesperado ao processar sua solicitação."
    
    def _execute_function_call(self, function_call: Dict[str, Any]) -> Result:
        """
        Executa uma function call específica.
        Wrapper para o sistema de ferramentas.
        """
        try:
            # Parsear function call
            parse_result = parse_function_call(function_call)
            if not parse_result.success:
                return parse_result
            
            tool_call = parse_result.data
            
            # Executar através do registry
            executed_call = self.tool_registry.execute_tool_call(tool_call, self.session_id)
            
            # Adicionar à thread
            self.current_thread = self.current_thread.add_tool_call(executed_call)
            
            if executed_call.status == "success":
                return Result.ok(executed_call.result)
            else:
                return Result.error(executed_call.error or "Erro desconhecido na execução da função")
                
        except Exception as e:
            return Result.error(f"Erro na execução da função: {str(e)}")
    
    def _generate_response_with_function_result(self, function_name: str, result: Any) -> str:
        """
        Gera resposta considerando o resultado de uma função executada.
        Simplificado para esta implementação.
        """
        return f"Executei a função '{function_name}' com sucesso. Resultado: {result}"
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calcula delay para retry com backoff exponencial opcional.
        Função pura de cálculo.
        """
        base_delay = self.config.retry_delay_seconds
        
        if self.config.exponential_backoff:
            return base_delay * (2 ** attempt)
        else:
            return base_delay
    
    def _save_to_memory(self) -> None:
        """
        Salva thread atual na memória.
        Efeito colateral controlado de persistência.
        """
        if not self.current_thread:
            return
        
        try:
            result = self.memory.save_thread(self.current_thread)
            if not result.success:
                self.logger.log_error(f"Erro ao salvar na memória: {result.error}")
        except Exception as e:
            self.logger.log_error(f"Erro inesperado ao salvar: {str(e)}")
    
    def _show_help(self) -> None:
        """Exibe ajuda com comandos disponíveis"""
        print("🆘 Comandos disponíveis:")
        print("  help, ajuda, h          - Mostra esta ajuda")
        print("  /stats                  - Estatísticas da sessão")
        print("  /tools                  - Lista ferramentas disponíveis")
        print("  /memory                 - Informações da memória")
        print("  /context                - Informações do contexto atual")
        print("  /clear                  - Limpa conversa atual")
        print("  /export [yaml|md|json]  - Exporta conversa")
        print("  quit, exit, sair, q     - Encerra o agente")
        print()
    
    def _handle_command(self, command: str) -> None:
        """Processa comandos especiais do usuário"""
        cmd = command.lower().strip()
        
        if cmd == "/stats":
            self._show_stats()
        elif cmd == "/tools":
            self._show_tools()
        elif cmd == "/memory":
            self._show_memory_info()
        elif cmd == "/context":
            self._show_context_info()
        elif cmd == "/clear":
            self._clear_conversation()
        elif cmd.startswith("/export"):
            parts = cmd.split()
            format_type = parts[1] if len(parts) > 1 else "yaml"
            self._export_conversation(format_type)
        else:
            print(f"❓ Comando desconhecido: {command}")
            print("Digite 'help' para ver comandos disponíveis.")
    
    def _show_stats(self) -> None:
        """Mostra estatísticas da sessão atual"""
        if not self.current_thread:
            print("📊 Nenhuma thread ativa.")
            return
        
        tools_stats = self.tool_registry.executor.get_stats()
        memory_stats_result = self.memory.get_stats()
        memory_stats = memory_stats_result.data if memory_stats_result.success else {}
        
        print("📊 Estatísticas da Sessão:")
        print(f"  🆔 ID da sessão: {self.session_id}")
        print(f"  💬 Total de mensagens: {len(self.current_thread.messages)}")
        print(f"  🔄 Turnos de conversa: {self.total_turns}")
        print(f"  🛠️  Ferramentas executadas: {len(self.current_thread.tools_calls)}")
        print(f"  🧠 Modelo usado: {self.config.model}")
        print(f"  💾 Ferramentas disponíveis: {tools_stats.get('total_tools', 0)}")
        print(f"  📚 Total de threads na memória: {memory_stats.get('total_threads', 0)}")
        print()
    
    def _show_tools(self) -> None:
        """Lista ferramentas disponíveis"""
        tools = self.tool_registry.get_available_tools()
        categories = self.tool_registry.executor.list_tools_by_category()
        
        print("🛠️  Ferramentas Disponíveis:")
        for category, tool_names in categories.items():
            print(f"  📁 {category.title()}:")
            for tool_name in tool_names:
                # Encontrar a ferramenta na lista completa
                tool_info = next((t for t in tools if t["name"] == tool_name), None)
                if tool_info:
                    print(f"    • {tool_name}: {tool_info['description']}")
        print()
    
    def _show_memory_info(self) -> None:
        """Mostra informações da memória"""
        stats_result = self.memory.get_stats()
        if stats_result.success:
            stats = stats_result.data
            print("💾 Informações da Memória:")
            print(f"  📊 Total de threads: {stats['total_threads']}")
            print(f"  💬 Total de mensagens: {stats['total_messages']}")
            print(f"  📝 Total de eventos: {stats['total_events']}")
            print(f"  📅 Última atividade: {stats.get('latest_activity', 'N/A')}")
            print(f"  💽 Tamanho do banco: {stats['db_size_mb']} MB")
        else:
            print(f"❌ Erro ao obter estatísticas: {stats_result.error}")
        print()
    
    def _show_context_info(self) -> None:
        """Mostra informações do contexto atual"""
        if not self.current_thread:
            print("📝 Nenhuma thread ativa.")
            return
        
        metadata = self.context_manager.builder.extract_context_metadata(self.current_thread)
        
        print("📝 Informações do Contexto:")
        print(f"  📊 Tokens estimados: {metadata.get('estimated_tokens', 0)}")
        print(f"  📏 Comprimento da conversa: {metadata.get('conversation_length', 0)}")
        print(f"  💬 Total de mensagens: {metadata.get('total_messages', 0)}")
        print(f"  ⏰ Minutos desde última mensagem: {metadata.get('time_since_last_message_minutes', 0):.1f}")
        print(f"  📈 Utilização da janela de contexto: {metadata.get('context_window_utilization', 0):.1%}")
        print()
    
    def _clear_conversation(self) -> None:
        """Limpa a conversa atual"""
        confirm = input("⚠️  Tem certeza que deseja limpar a conversa atual? (s/N): ").strip().lower()
        if confirm in ['s', 'sim', 'yes', 'y']:
            self.current_thread = Thread(session_id=f"session_{datetime.utcnow().timestamp()}")
            self.session_id = self.current_thread.session_id
            self.total_turns = 0
            print("✅ Conversa limpa. Nova sessão iniciada.")
        else:
            print("❌ Operação cancelada.")
        print()
    
    def _export_conversation(self, format_type: str) -> None:
        """Exporta a conversa atual"""
        if not self.current_thread:
            print("❌ Nenhuma conversa para exportar.")
            return
        
        try:
            from agent.context import ContextExporter
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type == "yaml":
                content = ContextExporter.to_yaml(self.current_thread)
                filename = f"conversa_{self.session_id}_{timestamp}.yaml"
            elif format_type == "md":
                content = ContextExporter.to_markdown(self.current_thread)
                filename = f"conversa_{self.session_id}_{timestamp}.md"
            elif format_type == "json":
                content = ContextExporter.to_json(self.current_thread)
                filename = f"conversa_{self.session_id}_{timestamp}.json"
            else:
                print(f"❌ Formato não suportado: {format_type}")
                return
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Conversa exportada para: {filename}")
            
        except Exception as e:
            print(f"❌ Erro ao exportar: {str(e)}")
        print()
    
    def _shutdown(self) -> None:
        """
        Encerra o agente graciosamente.
        Salva estado e limpa recursos.
        """
        self.is_running = False
        
        with log_operation(self.logger, "shutdown_agent", self.session_id):
            try:
                # Salvar estado final
                if self.current_thread and self.config.enable_memory_persistence:
                    self._save_to_memory()
                
                # Cleanup opcional
                if self.config.auto_cleanup_old_data:
                    cleanup_result = self.memory.cleanup_old_data(self.config.cleanup_days_threshold)
                    if cleanup_result.success:
                        self.logger.log_info("Cleanup automático realizado", **cleanup_result.data)
                
                # Log final
                self.logger.log_info(
                    "Agente encerrado graciosamente",
                    session_id=self.session_id,
                    total_turns=self.total_turns,
                    total_messages=len(self.current_thread.messages) if self.current_thread else 0
                )
                
                print("👋 Agente encerrado. Até logo!")
                
            except Exception as e:
                self.logger.log_error(f"Erro no shutdown: {str(e)}")
                print(f"⚠️  Aviso: Erro durante encerramento: {str(e)}")

# ============================================================================
# FACTORY FUNCTIONS E UTILITÁRIOS
# ============================================================================

def create_agent(config: Optional[AgentConfig] = None, session_id: Optional[str] = None) -> FunctionalAgent:
    """
    Factory function para criar instância do agente.
    Função pura de criação com configuração personalizada.
    """
    return FunctionalAgent(config, session_id)

def create_default_config() -> AgentConfig:
    """
    Cria configuração padrão do agente.
    Função pura de configuração.
    """
    return AgentConfig()

def load_config_from_env() -> AgentConfig:
    """
    Carrega configuração a partir de variáveis de ambiente.
    Função pura de configuração.
    """
    return AgentConfig(
        model=os.getenv("AGENT_MODEL", "mistralai/mistral-7b-instruct"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        api_base=os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1"),
        max_tokens=int(os.getenv("AGENT_MAX_TOKENS", "2000")),
        temperature=float(os.getenv("AGENT_TEMPERATURE", "0.7")),
        max_conversation_turns=int(os.getenv("AGENT_MAX_TURNS", "100")),
        enable_function_calling=os.getenv("AGENT_ENABLE_TOOLS", "true").lower() == "true",
        memory_db_path=os.getenv("AGENT_MEMORY_DB", "memory.db"),
        context_strategy=os.getenv("AGENT_CONTEXT_STRATEGY", "default"),
        max_retries=int(os.getenv("AGENT_MAX_RETRIES", "3"))
    )

def validate_environment() -> Result:
    """
    Valida se o ambiente está configurado corretamente.
    Função pura de validação.
    """
    errors = []
    warnings = []
    
    # Verificar API key
    if not os.getenv("OPENROUTER_API_KEY"):
        errors.append("OPENROUTER_API_KEY não configurada")
    
    # Verificar dependências opcionais
    try:
        import yaml
    except ImportError:
        warnings.append("PyYAML não instalado - funcionalidades YAML limitadas")
    
    # Verificar permissões de escrita
    try:
        test_file = "test_write_permission.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except Exception:
        warnings.append("Permissões de escrita limitadas - logging pode ser afetado")
    
    if errors:
        return Result.error({
            "errors": errors,
            "warnings": warnings,
            "valid": False
        })
    
    return Result.ok({
        "errors": [],
        "warnings": warnings,
        "valid": True
    })

# ============================================================================
# AGENTE ASSÍNCRONO (EXTENSÃO FUTURA)
# ============================================================================

class AsyncFunctionalAgent:
    """
    Versão assíncrona do agente para uso em aplicações web/API.
    Implementação futura para escalabilidade.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.logger = get_logger()
        # TODO: Implementar versão assíncrona completa
    
    async def process_user_message_async(self, message: str, session_id: str) -> Result:
        """
        Versão assíncrona do processamento de mensagens.
        Para implementação futura.
        """
        # TODO: Implementar lógica assíncrona
        return Result.error("Implementação assíncrona ainda não disponível")
    
    async def start_conversation_stream(self, session_id: str) -> Generator[str, None, None]:
        """
        Stream assíncrono de conversa.
        Para implementação futura com WebSockets/SSE.
        """
        # TODO: Implementar streaming assíncrono
        yield "Streaming não implementado ainda"

# ============================================================================
# SISTEMA DE MIDDLEWARE (EXTENSIBILIDADE)
# ============================================================================

class AgentMiddleware:
    """
    Classe base para middleware do agente.
    Permite interceptar e modificar fluxo de processamento.
    """
    
    def before_user_message(self, message: str, session_id: str) -> Result:
        """Hook chamado antes de processar mensagem do usuário"""
        return Result.ok(message)
    
    def after_user_message(self, message: str, response: str, session_id: str) -> Result:
        """Hook chamado após processar mensagem do usuário"""
        return Result.ok(response)
    
    def before_llm_call(self, messages: List[Dict[str, Any]], session_id: str) -> Result:
        """Hook chamado antes da chamada LLM"""
        return Result.ok(messages)
    
    def after_llm_call(self, response: Dict[str, Any], session_id: str) -> Result:
        """Hook chamado após chamada LLM"""
        return Result.ok(response)
    
    def on_tool_execution(self, tool_name: str, arguments: Dict[str, Any], 
                         result: Any, session_id: str) -> None:
        """Hook chamado após execução de ferramenta"""
        pass
    
    def on_error(self, error: Exception, context: str, session_id: str) -> None:
        """Hook chamado quando ocorre erro"""
        pass

class ContentFilterMiddleware(AgentMiddleware):
    """
    Middleware para filtrar conteúdo inadequado.
    Exemplo de implementação de middleware.
    """
    
    def __init__(self, blocked_words: List[str] = None):
        self.blocked_words = blocked_words or []
        self.logger = get_logger()
    
    def before_user_message(self, message: str, session_id: str) -> Result:
        """Filtra conteúdo da mensagem do usuário"""
        message_lower = message.lower()
        
        for blocked_word in self.blocked_words:
            if blocked_word.lower() in message_lower:
                self.logger.log_info(
                    f"Conteúdo bloqueado detectado: {blocked_word}",
                    session_id=session_id
                )
                return Result.error(f"Mensagem contém conteúdo não permitido")
        
        return Result.ok(message)
    
    def after_user_message(self, message: str, response: str, session_id: str) -> Result:
        """Filtra conteúdo da resposta do agente"""
        response_lower = response.lower()
        
        for blocked_word in self.blocked_words:
            if blocked_word.lower() in response_lower:
                self.logger.log_info(
                    f"Resposta bloqueada por conteúdo: {blocked_word}",
                    session_id=session_id
                )
                return Result.ok("Desculpe, não posso fornecer essa informação.")
        
        return Result.ok(response)

class RateLimitMiddleware(AgentMiddleware):
    """
    Middleware para controlar taxa de requisições.
    Exemplo de implementação de middleware.
    """
    
    def __init__(self, max_requests_per_minute: int = 30):
        self.max_requests = max_requests_per_minute
        self.requests_log: Dict[str, List[datetime]] = {}
        self.logger = get_logger()
    
    def before_user_message(self, message: str, session_id: str) -> Result:
        """Verifica rate limit antes de processar mensagem"""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        
        # Limpar requisições antigas
        if session_id in self.requests_log:
            self.requests_log[session_id] = [
                req_time for req_time in self.requests_log[session_id]
                if req_time > cutoff
            ]
        else:
            self.requests_log[session_id] = []
        
        # Verificar limite
        current_requests = len(self.requests_log[session_id])
        
        if current_requests >= self.max_requests:
            self.logger.log_info(
                f"Rate limit excedido para sessão {session_id}",
                current_requests=current_requests,
                max_requests=self.max_requests
            )
            return Result.error(f"Muitas requisições. Limite: {self.max_requests}/minuto")
        
        # Registrar nova requisição
        self.requests_log[session_id].append(now)
        return Result.ok(message)

# ============================================================================
# AGENT BUILDER COM FLUENT INTERFACE
# ============================================================================

class AgentBuilder:
    """
    Builder pattern para configurar agente de forma fluente.
    Facilita configuração complexa e extensível.
    """
    
    def __init__(self):
        self.config = AgentConfig()
        self.middlewares: List[AgentMiddleware] = []
    
    def with_model(self, model: str) -> 'AgentBuilder':
        """Configura modelo LLM"""
        self.config = AgentConfig(
            **{**self.config.__dict__, 'model': model}
        )
        return self
    
    def with_api_key(self, api_key: str) -> 'AgentBuilder':
        """Configura API key"""
        self.config = AgentConfig(
            **{**self.config.__dict__, 'api_key': api_key}
        )
        return self
    
    def with_temperature(self, temperature: float) -> 'AgentBuilder':
        """Configura temperatura do modelo"""
        self.config = AgentConfig(
            **{**self.config.__dict__, 'temperature': temperature}
        )
        return self
    
    def with_max_tokens(self, max_tokens: int) -> 'AgentBuilder':
        """Configura máximo de tokens"""
        self.config = AgentConfig(
            **{**self.config.__dict__, 'max_tokens': max_tokens}
        )
        return self
    
    def with_memory_persistence(self, enabled: bool = True, db_path: str = "memory.db") -> 'AgentBuilder':
        """Configura persistência de memória"""
        self.config = AgentConfig(
            **{**self.config.__dict__, 
               'enable_memory_persistence': enabled,
               'memory_db_path': db_path}
        )
        return self
    
    def with_function_calling(self, enabled: bool = True) -> 'AgentBuilder':
        """Configura tool use/function calling"""
        self.config = AgentConfig(
            **{**self.config.__dict__, 'enable_function_calling': enabled}
        )
        return self
    
    def with_context_strategy(self, strategy: str) -> 'AgentBuilder':
        """Configura estratégia de contexto"""
        self.config = AgentConfig(
            **{**self.config.__dict__, 'context_strategy': strategy}
        )
        return self
    
    def with_retry_config(self, max_retries: int = 3, delay: float = 1.0, 
                         exponential_backoff: bool = True) -> 'AgentBuilder':
        """Configura política de retry"""
        self.config = AgentConfig(
            **{**self.config.__dict__, 
               'max_retries': max_retries,
               'retry_delay_seconds': delay,
               'exponential_backoff': exponential_backoff}
        )
        return self
    
    def with_middleware(self, middleware: AgentMiddleware) -> 'AgentBuilder':
        """Adiciona middleware"""
        self.middlewares.append(middleware)
        return self
    
    def with_content_filter(self, blocked_words: List[str]) -> 'AgentBuilder':
        """Adiciona filtro de conteúdo"""
        return self.with_middleware(ContentFilterMiddleware(blocked_words))
    
    def with_rate_limit(self, requests_per_minute: int = 30) -> 'AgentBuilder':
        """Adiciona rate limiting"""
        return self.with_middleware(RateLimitMiddleware(requests_per_minute))
    
    def build(self, session_id: Optional[str] = None) -> FunctionalAgent:
        """Constrói o agente com configurações definidas"""
        agent = FunctionalAgent(self.config, session_id)
        
        # TODO: Integrar middlewares ao agente
        # Esta funcionalidade seria implementada como uma extensão
        if self.middlewares:
            agent.logger.log_info(f"Agente criado com {len(self.middlewares)} middlewares")
        
        return agent

# ============================================================================
# UTILITÁRIOS DE CONVENIÊNCIA
# ============================================================================

def quick_start(model: str = "mistralai/mistral-7b-instruct", 
               session_id: Optional[str] = None) -> FunctionalAgent:
    """
    Início rápido com configurações padrão.
    Função de conveniência para uso simples.
    """
    return (AgentBuilder()
            .with_model(model)
            .with_api_key(os.getenv("OPENROUTER_API_KEY"))
            .build(session_id))

def create_production_agent(session_id: Optional[str] = None) -> FunctionalAgent:
    """
    Cria agente para produção com configurações robustas.
    Função de conveniência para ambientes de produção.
    """
    return (AgentBuilder()
            .with_model("openai/gpt-4")  # Modelo mais robusto
            .with_temperature(0.3)  # Mais determinístico
            .with_retry_config(max_retries=5, exponential_backoff=True)
            .with_memory_persistence(enabled=True)
            .with_function_calling(enabled=True)
            .with_context_strategy("compressed")
            .with_rate_limit(requests_per_minute=20)  # Rate limit conservador
            .build(session_id))

def create_development_agent(session_id: Optional[str] = None) -> FunctionalAgent:
    """
    Cria agente para desenvolvimento com logs verbosos.
    Função de conveniência para desenvolvimento.
    """
    return (AgentBuilder()
            .with_model("mistralai/mistral-7b-instruct")  # Modelo mais barato
            .with_temperature(0.7)
            .with_memory_persistence(enabled=True, db_path="dev_memory.db")
            .with_function_calling(enabled=True)
            .with_context_strategy("default")
            .build(session_id))

# ============================================================================
# FUNÇÃO PRINCIPAL PARA EXECUÇÃO DIRETA
# ============================================================================

def main_agent():
    """
    Função principal para execução direta do agente.
    Compatibilidade com o código legado.
    """
    try:
        # Validar ambiente
        env_validation = validate_environment()
        if not env_validation.success:
            print("❌ Erro na validação do ambiente:")
            for error in env_validation.data["errors"]:
                print(f"  • {error}")
            return
        
        # Exibir warnings se houver
        if env_validation.data.get("warnings"):
            print("⚠️  Avisos:")
            for warning in env_validation.data["warnings"]:
                print(f"  • {warning}")
            print()
        
        # Criar e iniciar agente
        agent = quick_start()
        agent.start_conversation()
        
    except KeyboardInterrupt:
        print("\n🛑 Interrompido pelo usuário.")
    except Exception as e:
        print(f"❌ Erro fatal: {str(e)}")
        get_logger().log_error(f"Erro fatal na execução principal: {str(e)}")

if __name__ == "__main__":
    main_agent()
