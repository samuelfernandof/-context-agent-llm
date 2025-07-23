"""
Agent Package - Sistema de Agente Funcional
===========================================

Este pacote implementa um agente inteligente funcional com:
- Memória persistente usando SQLite
- Sistema de logging estruturado
- Gerenciamento inteligente de contexto
- Tool use via OpenAI Function Calls
- Arquitetura funcional e imutável

Componentes principais:
- agent.py: Loop principal do agente
- memory.py: Sistema de memória persistente
- context.py: Gerenciamento de contexto com YAML
- tools.py: Sistema de ferramentas/funções
- logger.py: Logging estruturado

Exemplo de uso básico:
    >>> from agent import FunctionalAgent, AgentConfig
    >>> config = AgentConfig(model="mistralai/mistral-7b-instruct")
    >>> agent = FunctionalAgent(config)
    >>> agent.start_conversation()

Exemplo de uso avançado:
    >>> from agent import AgentBuilder
    >>> agent = (AgentBuilder()
    ...          .with_model("openai/gpt-4")
    ...          .with_temperature(0.3)
    ...          .with_memory_persistence(True)
    ...          .with_function_calling(True)
    ...          .build())
    >>> agent.start_conversation()
"""

# ============================================================================
# VERSÃO E METADADOS
# ============================================================================

__version__ = "1.0.0"
__author__ = "Agente Funcional Team"
__email__ = "contact@agente-funcional.com"
__description__ = "Sistema de Agente Inteligente Funcional com Memória Persistente"
__url__ = "https://github.com/user/context-agent-llm"

# ============================================================================
# IMPORTAÇÕES PRINCIPAIS
# ============================================================================

# Classes principais do agente
from .agent import (
    FunctionalAgent,
    AgentConfig,
    AgentBuilder,
    AsyncFunctionalAgent,
    AgentMiddleware,
    ContentFilterMiddleware,
    RateLimitMiddleware
)

# Sistema de memória
from .memory import (
    AgentMemory,
    get_memory,
    create_empty_thread,
    backup_memory
)

# Sistema de contexto
from .context import (
    ContextBuilder,
    ContextManager,
    ContextFilter,
    ContextExporter,
    ContextValidator,
    ContextCache,
    create_context_manager,
    build_context,
    export_thread_as_yaml,
    export_thread_as_markdown,
    validate_thread_integrity,
    get_context_cache
)

# Sistema de ferramentas
from .tools import (
    ToolExecutor,
    ToolRegistry,
    ToolMetadata,
    ToolParameter,
    ToolPlugin,
    tool,
    parameter,
    get_tool_registry,
    create_tool_call,
    parse_function_call,
    get_tools,
    load_plugin
)

# Sistema de logging
from .logger import (
    StructuredLogger,
    LogAnalyzer,
    get_logger,
    log_event,
    log_info,
    log_error,
    log_operation,
    create_event
)

# ============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# ============================================================================

def quick_start(model: str = "mistralai/mistral-7b-instruct", 
               session_id: str = None) -> FunctionalAgent:
    """
    Início rápido com configurações padrão.
    
    Args:
        model: Modelo LLM a usar
        session_id: ID da sessão (opcional)
    
    Returns:
        FunctionalAgent configurado e pronto para uso
    
    Example:
        >>> agent = quick_start("openai/gpt-3.5-turbo")
        >>> agent.start_conversation()
    """
    from .agent import quick_start as _quick_start
    return _quick_start(model, session_id)

def create_production_agent(session_id: str = None) -> FunctionalAgent:
    """
    Cria agente otimizado para produção.
    
    Args:
        session_id: ID da sessão (opcional)
    
    Returns:
        FunctionalAgent com configurações robustas
    
    Example:
        >>> agent = create_production_agent()
        >>> agent.start_conversation()
    """
    from .agent import create_production_agent as _create_prod
    return _create_prod(session_id)

def create_development_agent(session_id: str = None) -> FunctionalAgent:
    """
    Cria agente otimizado para desenvolvimento.
    
    Args:
        session_id: ID da sessão (opcional)
    
    Returns:
        FunctionalAgent com configurações de desenvolvimento
    
    Example:
        >>> agent = create_development_agent()
        >>> agent.start_conversation()
    """
    from .agent import create_development_agent as _create_dev
    return _create_dev(session_id)

def validate_environment():
    """
    Valida se o ambiente está configurado corretamente.
    
    Returns:
        Result com informações de validação
    
    Example:
        >>> result = validate_environment()
        >>> if result.success:
        ...     print("Ambiente OK!")
        >>> else:
        ...     print(f"Problemas: {result.data['errors']}")
    """
    from .agent import validate_environment as _validate
    return _validate()

# ============================================================================
# CONFIGURAÇÃO DE LOGGING PADRÃO
# ============================================================================

def setup_logging(level: str = "INFO", log_file: str = "agent.log"):
    """
    Configura logging padrão para o pacote.
    
    Args:
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Arquivo de log
    
    Example:
        >>> setup_logging("DEBUG", "debug.log")
    """
    import logging
    
    # Converter string para nível
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configurar logger global
    logger = get_logger()
    logger.logger.setLevel(numeric_level)
    
    # Log de configuração
    logger.log_info(f"Logging configurado: nível {level}, arquivo {log_file}")

# ============================================================================
# UTILITÁRIOS DE CONFIGURAÇÃO
# ============================================================================

def load_config_from_file(config_path: str):
    """
    Carrega configuração de arquivo JSON.
    
    Args:
        config_path: Caminho para arquivo de configuração
    
    Returns:
        Result com AgentConfig ou erro
    
    Example:
        >>> result = load_config_from_file("config.json")
        >>> if result.success:
        ...     agent = FunctionalAgent(result.data)
    """
    import json
    from pathlib import Path
    from models.models import Result
    
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            return Result.error(f"Arquivo não encontrado: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        config = AgentConfig(**config_data)
        return Result.ok(config)
        
    except Exception as e:
        return Result.error(f"Erro ao carregar configuração: {str(e)}")

def create_default_config():
    """
    Cria configuração padrão.
    
    Returns:
        AgentConfig com valores padrão
    
    Example:
        >>> config = create_default_config()
        >>> config.model = "openai/gpt-4"
        >>> agent = FunctionalAgent(config)
    """
    return AgentConfig()

# ============================================================================
# EXPORTAÇÕES PARA COMPATIBILIDADE
# ============================================================================

# Aliases para compatibilidade com versões anteriores
Agent = FunctionalAgent
Config = AgentConfig
Builder = AgentBuilder

# Funções de compatibilidade
def create_agent(config=None, session_id=None):
    """Alias para FunctionalAgent(config, session_id)"""
    return FunctionalAgent(config, session_id)

def get_default_tools():
    """Alias para get_tools()"""
    return get_tools()

# ============================================================================
# LISTA DE EXPORTAÇÕES PÚBLICAS
# ============================================================================

__all__ = [
    # Classes principais
    "FunctionalAgent",
    "AgentConfig", 
    "AgentBuilder",
    "AsyncFunctionalAgent",
    
    # Middleware
    "AgentMiddleware",
    "ContentFilterMiddleware",
    "RateLimitMiddleware",
    
    # Memória
    "AgentMemory",
    "get_memory",
    "create_empty_thread",
    "backup_memory",
    
    # Contexto
    "ContextBuilder",
    "ContextManager",
    "ContextFilter",
    "ContextExporter",
    "ContextValidator",
    "ContextCache",
    "create_context_manager",
    "build_context",
    "export_thread_as_yaml",
    "export_thread_as_markdown",
    "validate_thread_integrity",
    "get_context_cache",
    
    # Ferramentas
    "ToolExecutor",
    "ToolRegistry",
    "ToolMetadata",
    "ToolParameter",
    "ToolPlugin",
    "tool",
    "parameter",
    "get_tool_registry",
    "create_tool_call",
    "parse_function_call",
    "get_tools",
    "load_plugin",
    
    # Logging
    "StructuredLogger",
    "LogAnalyzer",
    "get_logger",
    "log_event",
    "log_info",
    "log_error",
    "log_operation",
    "create_event",
    "setup_logging",
    
    # Funções de conveniência
    "quick_start",
    "create_production_agent",
    "create_development_agent",
    "validate_environment",
    "load_config_from_file",
    "create_default_config",
    
    # Aliases de compatibilidade
    "Agent",
    "Config",
    "Builder",
    "create_agent",
    "get_default_tools",
    
    # Metadados
    "__version__",
    "__author__",
    "__description__"
]

# ============================================================================
# INICIALIZAÇÃO AUTOMÁTICA
# ============================================================================

# Log de inicialização do pacote
try:
    _logger = get_logger()
    _logger.log_info(
        f"Pacote agent inicializado",
        version=__version__,
        components=len(__all__)
    )
except Exception:
    # Falha silenciosa se logger não estiver disponível
    pass
