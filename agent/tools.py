import json
import uuid
import inspect
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from functools import wraps
import traceback

from models.models import ToolCall, Result, Event
from agent.logger import get_logger, log_operation

# ============================================================================
# TIPOS E METADADOS DE FERRAMENTAS
# ============================================================================

@dataclass(frozen=True)
class ToolParameter:
    """
    Representa um parâmetro de ferramenta de forma imutável.
    Compatível com OpenAI Function Calling schema.
    """
    name: str
    type: str  # "string", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """Converte para schema JSON compatível com OpenAI"""
        schema = {
            "type": self.type,
            "description": self.description
        }
        
        if self.enum:
            schema["enum"] = self.enum
        
        if self.default is not None:
            schema["default"] = self.default
            
        return schema

@dataclass(frozen=True)
class ToolMetadata:
    """
    Metadados imutáveis de uma ferramenta.
    Define interface e comportamento da função.
    """
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    category: str = "general"
    version: str = "1.0.0"
    requires_confirmation: bool = False
    is_dangerous: bool = False
    max_retries: int = 3
    timeout_seconds: int = 30
    
    def to_openai_schema(self) -> Dict[str, Any]:
        """
        Converte para schema OpenAI Function Calling.
        Função pura de transformação.
        """
        required_params = [p.name for p in self.parameters if p.required]
        properties = {p.name: p.to_schema() for p in self.parameters}
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_params
            }
        }
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> Result:
        """
        Valida argumentos contra os parâmetros definidos.
        Função pura de validação.
        """
        errors = []
        
        # Verificar parâmetros obrigatórios
        required_params = {p.name for p in self.parameters if p.required}
        provided_params = set(arguments.keys())
        
        missing_params = required_params - provided_params
        if missing_params:
            errors.append(f"Parâmetros obrigatórios faltando: {', '.join(missing_params)}")
        
        # Verificar parâmetros extras
        valid_params = {p.name for p in self.parameters}
        extra_params = provided_params - valid_params
        if extra_params:
            errors.append(f"Parâmetros desconhecidos: {', '.join(extra_params)}")
        
        # Validar tipos e valores
        for param in self.parameters:
            if param.name in arguments:
                value = arguments[param.name]
                validation_error = self._validate_parameter_value(param, value)
                if validation_error:
                    errors.append(f"Parâmetro '{param.name}': {validation_error}")
        
        if errors:
            return Result.error("; ".join(errors))
        
        return Result.ok("Argumentos válidos")
    
    def _validate_parameter_value(self, param: ToolParameter, value: Any) -> Optional[str]:
        """
        Valida valor individual de parâmetro.
        Função auxiliar pura.
        """
        # Validação de enum
        if param.enum and value not in param.enum:
            return f"deve ser um dos valores: {param.enum}"
        
        # Validação básica de tipo
        if param.type == "string" and not isinstance(value, str):
            return "deve ser uma string"
        elif param.type == "number" and not isinstance(value, (int, float)):
            return "deve ser um número"
        elif param.type == "boolean" and not isinstance(value, bool):
            return "deve ser um boolean"
        elif param.type == "array" and not isinstance(value, list):
            return "deve ser um array"
        elif param.type == "object" and not isinstance(value, dict):
            return "deve ser um objeto"
        
        return None

# ============================================================================
# DECORATORS PARA REGISTRO DE FERRAMENTAS
# ============================================================================

def tool(name: Optional[str] = None,
         description: str = "",
         category: str = "general",
         requires_confirmation: bool = False,
         is_dangerous: bool = False,
         max_retries: int = 3,
         timeout_seconds: int = 30):
    """
    Decorator para registrar funções como ferramentas do agente.
    Extrai automaticamente metadados da função.
    
    Uso:
    @tool(name="calcular", description="Realiza cálculos matemáticos")
    def calculate(expression: str) -> float:
        return eval(expression)
    """
    def decorator(func: Callable) -> Callable:
        # Extrair nome da função se não fornecido
        tool_name = name or func.__name__
        
        # Extrair parâmetros da assinatura da função
        sig = inspect.signature(func)
        parameters = []
        
        for param_name, param in sig.parameters.items():
            # Determinar tipo do parâmetro
            param_type = "string"  # padrão
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int or param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
                elif param.annotation == dict:
                    param_type = "object"
            
            # Determinar se é obrigatório
            required = param.default == inspect.Parameter.empty
            
            # Criar parâmetro
            tool_param = ToolParameter(
                name=param_name,
                type=param_type,
                description=f"Parâmetro {param_name}",
                required=required,
                default=param.default if not required else None
            )
            parameters.append(tool_param)
        
        # Criar metadados
        metadata = ToolMetadata(
            name=tool_name,
            description=description or func.__doc__ or "Ferramenta sem descrição",
            parameters=parameters,
            category=category,
            requires_confirmation=requires_confirmation,
            is_dangerous=is_dangerous,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds
        )
        
        # Adicionar metadados à função
        func._tool_metadata = metadata
        func._is_tool = True
        
        return func
    
    return decorator

def parameter(name: str, description: str, param_type: str = "string", 
              required: bool = True, enum: Optional[List[Any]] = None):
    """
    Decorator para documentar parâmetros de ferramentas mais detalhadamente.
    Usado junto com @tool para maior controle.
    
    Uso:
    @tool(name="search")
    @parameter("query", "Termo de busca", "string", True)
    @parameter("limit", "Número máximo de resultados", "number", False)
    def search_web(query: str, limit: int = 10):
        pass
    """
    def decorator(func: Callable) -> Callable:
        if not hasattr(func, '_tool_parameters'):
            func._tool_parameters = []
        
        param = ToolParameter(
            name=name,
            type=param_type,
            description=description,
            required=required,
            enum=enum
        )
        func._tool_parameters.append(param)
        
        return func
    
    return decorator

# ============================================================================
# EXECUTOR DE FERRAMENTAS
# ============================================================================

class ToolExecutor:
    """
    Executor thread-safe para ferramentas do agente.
    Gerencia execução, retries, timeouts e logging.
    """
    
    def __init__(self):
        self.logger = get_logger()
        self._registered_tools: Dict[str, Callable] = {}
        self._tool_metadata: Dict[str, ToolMetadata] = {}
    
    def register_tool(self, func: Callable, metadata: Optional[ToolMetadata] = None) -> Result:
        """
        Registra uma ferramenta para uso pelo agente.
        Efeito colateral controlado de registro.
        """
        try:
            # Usar metadados do decorator se disponível
            if hasattr(func, '_tool_metadata'):
                tool_metadata = func._tool_metadata
            elif metadata:
                tool_metadata = metadata
            else:
                return Result.error(f"Função {func.__name__} não possui metadados de ferramenta")
            
            # Validar se não há conflito de nomes
            if tool_metadata.name in self._registered_tools:
                return Result.error(f"Ferramenta '{tool_metadata.name}' já registrada")
            
            # Registrar
            self._registered_tools[tool_metadata.name] = func
            self._tool_metadata[tool_metadata.name] = tool_metadata
            
            self.logger.log_info(
                f"Ferramenta registrada: {tool_metadata.name}",
                tool_name=tool_metadata.name,
                category=tool_metadata.category,
                parameters_count=len(tool_metadata.parameters)
            )
            
            return Result.ok(f"Ferramenta '{tool_metadata.name}' registrada com sucesso")
            
        except Exception as e:
            error_msg = f"Erro ao registrar ferramenta: {str(e)}"
            self.logger.log_error(error_msg)
            return Result.error(error_msg)
    
    def execute_tool(self, tool_call: ToolCall, session_id: Optional[str] = None) -> ToolCall:
        """
        Executa uma ferramenta e retorna ToolCall atualizado.
        Função com efeitos colaterais controlados (execução + logging).
        """
        start_time = datetime.utcnow()
        
        # Log início da execução
        self.logger.log_function_call(
            tool_call.name, 
            tool_call.arguments, 
            session_id
        )
        
        try:
            with log_operation(self.logger, f"execute_tool_{tool_call.name}", session_id):
                # Verificar se ferramenta está registrada
                if tool_call.name not in self._registered_tools:
                    return self._create_error_result(
                        tool_call, 
                        f"Ferramenta '{tool_call.name}' não encontrada"
                    )
                
                func = self._registered_tools[tool_call.name]
                metadata = self._tool_metadata[tool_call.name]
                
                # Validar argumentos
                validation_result = metadata.validate_arguments(tool_call.arguments)
                if not validation_result.success:
                    return self._create_error_result(tool_call, validation_result.error)
                
                # Verificar se requer confirmação (em implementação real)
                if metadata.requires_confirmation:
                    self.logger.log_info(
                        f"Ferramenta '{tool_call.name}' requer confirmação (auto-aprovado para demo)",
                        session_id=session_id
                    )
                
                # Executar com retry logic
                result = self._execute_with_retry(func, tool_call.arguments, metadata)
                
                # Criar ToolCall de sucesso
                success_tool_call = ToolCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                    status="success",
                    result=result,
                    timestamp=start_time
                )
                
                # Log resultado
                self.logger.log_function_result(
                    tool_call.name,
                    result,
                    True,
                    session_id
                )
                
                return success_tool_call
                
        except Exception as e:
            error_msg = f"Erro na execução: {str(e)}"
            self.logger.log_error(error_msg, session_id=session_id, tool_name=tool_call.name)
            return self._create_error_result(tool_call, error_msg)
    
    def _execute_with_retry(self, func: Callable, arguments: Dict[str, Any], 
                           metadata: ToolMetadata) -> Any:
        """
        Executa função com lógica de retry.
        Função interna com efeitos colaterais.
        """
        last_exception = None
        
        for attempt in range(metadata.max_retries):
            try:
                # TODO: Implementar timeout real com asyncio/threading
                result = func(**arguments)
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < metadata.max_retries - 1:
                    self.logger.log_info(
                        f"Tentativa {attempt + 1} falhou, tentando novamente",
                        error=str(e),
                        tool_name=metadata.name
                    )
                    # Em implementação real, adicionar delay exponential backoff
                    continue
                else:
                    break
        
        # Se chegou aqui, todas as tentativas falharam
        raise last_exception or Exception("Execução falhou após todas as tentativas")
    
    def _create_error_result(self, original_call: ToolCall, error_message: str) -> ToolCall:
        """
        Cria ToolCall de erro.
        Função auxiliar pura.
        """
        return ToolCall(
            id=original_call.id,
            name=original_call.name,
            arguments=original_call.arguments,
            status="error",
            error=error_message,
            timestamp=original_call.timestamp
        )
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Retorna lista de ferramentas disponíveis no formato OpenAI.
        Função read-only pura.
        """
        return [metadata.to_openai_schema() for metadata in self._tool_metadata.values()]
    
    def get_tool_metadata(self, tool_name: str) -> Optional[ToolMetadata]:
        """
        Retorna metadados de uma ferramenta específica.
        Função read-only pura.
        """
        return self._tool_metadata.get(tool_name)
    
    def list_tools_by_category(self) -> Dict[str, List[str]]:
        """
        Agrupa ferramentas por categoria.
        Função read-only pura.
        """
        categories = {}
        for name, metadata in self._tool_metadata.items():
            if metadata.category not in categories:
                categories[metadata.category] = []
            categories[metadata.category].append(name)
        
        return categories
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas das ferramentas registradas.
        Função read-only pura.
        """
        total_tools = len(self._registered_tools)
        categories = self.list_tools_by_category()
        dangerous_tools = sum(1 for m in self._tool_metadata.values() if m.is_dangerous)
        
        return {
            "total_tools": total_tools,
            "categories": categories,
            "dangerous_tools": dangerous_tools,
            "tools_by_category": {cat: len(tools) for cat, tools in categories.items()}
        }

# ============================================================================
# FERRAMENTAS BÁSICAS PRÉ-DEFINIDAS
# ============================================================================

@tool(
    name="get_current_time",
    description="Obtém a data e hora atual",
    category="utilities"
)
def get_current_time() -> str:
    """Retorna data e hora atual formatada"""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

@tool(
    name="calculate",
    description="Realiza cálculos matemáticos simples",
    category="math",
    is_dangerous=True  # eval pode ser perigoso
)
def calculate(expression: str) -> Union[float, str]:
    """
    Calcula expressões matemáticas simples.
    ATENÇÃO: Usa eval() - apenas para demonstração!
    """
    try:
        # Lista de operações permitidas (segurança básica)
        allowed_chars = set('0123456789+-*/()., ')
        if not all(c in allowed_chars for c in expression):
            return "Erro: Expressão contém caracteres não permitidos"
        
        # Evitar algumas funções perigosas
        dangerous_keywords = ['import', 'exec', 'eval', '__', 'open', 'file']
        if any(keyword in expression.lower() for keyword in dangerous_keywords):
            return "Erro: Expressão contém operações não permitidas"
        
        result = eval(expression)
        return float(result) if isinstance(result, (int, float)) else str(result)
    
    except Exception as e:
        return f"Erro no cálculo: {str(e)}"

@tool(
    name="echo",
    description="Repete o texto fornecido (útil para testes)",
    category="utilities"
)
def echo(text: str, repeat: int = 1) -> str:
    """Ecoa o texto especificado, opcionalmente múltiplas vezes"""
    if repeat < 1 or repeat > 10:
        return "Erro: repeat deve estar entre 1 e 10"
    
    return " | ".join([text] * repeat)

@tool(
    name="count_words",
    description="Conta palavras em um texto",
    category="text_processing"
)
def count_words(text: str) -> Dict[str, Any]:
    """Conta palavras, caracteres e linhas em um texto"""
    words = text.split()
    return {
        "word_count": len(words),
        "character_count": len(text),
        "character_count_no_spaces": len(text.replace(" ", "")),
        "line_count": len(text.split("\n"))
    }

@tool(
    name="format_json",
    description="Formata uma string JSON para melhor legibilidade",
    category="utilities"
)
def format_json(json_string: str, indent: int = 2) -> str:
    """Formata JSON com indentação legível"""
    try:
        parsed = json.loads(json_string)
        return json.dumps(parsed, indent=indent, ensure_ascii=False)
    except json.JSONDecodeError as e:
        return f"Erro: JSON inválido - {str(e)}"

@tool(
    name="generate_uuid",
    description="Gera um UUID único",
    category="utilities"
)
def generate_uuid(version: int = 4) -> str:
    """Gera UUID nas versões 1 ou 4"""
    if version == 1:
        return str(uuid.uuid1())
    elif version == 4:
        return str(uuid.uuid4())
    else:
        return "Erro: Apenas versões 1 e 4 são suportadas"

# ============================================================================
# TOOL REGISTRY E DESCOBERTA AUTOMÁTICA
# ============================================================================

class ToolRegistry:
    """
    Registry central para descoberta e registro automático de ferramentas.
    Implementa padrão singleton para acesso global.
    """
    
    _instance: Optional['ToolRegistry'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.executor = ToolExecutor()
        self.logger = get_logger()
        self._auto_register_builtin_tools()
        self._initialized = True
    
    def _auto_register_builtin_tools(self) -> None:
        """
        Registra automaticamente ferramentas pré-definidas.
        Efeito colateral de inicialização.
        """
        # Encontrar todas as funções decoradas com @tool neste módulo
        current_module = globals()
        
        for name, obj in current_module.items():
            if callable(obj) and hasattr(obj, '_is_tool'):
                result = self.executor.register_tool(obj)
                if not result.success:
                    self.logger.log_error(f"Falha ao registrar ferramenta {name}: {result.error}")
    
    def register_tool(self, func: Callable, metadata: Optional[ToolMetadata] = None) -> Result:
        """Wrapper para registro de ferramentas"""
        return self.executor.register_tool(func, metadata)
    
    def execute_tool_call(self, tool_call: ToolCall, session_id: Optional[str] = None) -> ToolCall:
        """Wrapper para execução de ferramentas"""
        return self.executor.execute_tool(tool_call, session_id)
    
    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Wrapper para obter ferramentas disponíveis"""
        return self.executor.get_available_tools()
    
    def discover_tools_in_module(self, module_name: str) -> Result:
        """
        Descobre e registra ferramentas em um módulo específico.
        Útil para plugins externos.
        """
        try:
            import importlib
            module = importlib.import_module(module_name)
            
            registered_count = 0
            for name in dir(module):
                obj = getattr(module, name)
                if callable(obj) and hasattr(obj, '_is_tool'):
                    result = self.register_tool(obj)
                    if result.success:
                        registered_count += 1
                    else:
                        self.logger.log_error(f"Falha ao registrar {name}: {result.error}")
            
            return Result.ok(f"Registradas {registered_count} ferramentas do módulo {module_name}")
            
        except Exception as e:
            error_msg = f"Erro ao descobrir ferramentas em {module_name}: {str(e)}"
            self.logger.log_error(error_msg)
            return Result.error(error_msg)

# ============================================================================
# UTILITÁRIOS E FACTORY FUNCTIONS
# ============================================================================

def get_tool_registry() -> ToolRegistry:
    """
    Retorna instância global do registry de ferramentas.
    Padrão singleton thread-safe.
    """
    return ToolRegistry()

def create_tool_call(tool_name: str, arguments: Dict[str, Any], 
                    call_id: Optional[str] = None) -> ToolCall:
    """
    Factory function para criar ToolCall.
    Função pura de criação.
    """
    return ToolCall(
        id=call_id or str(uuid.uuid4()),
        name=tool_name,
        arguments=arguments,
        timestamp=datetime.utcnow()
    )

def parse_function_call(function_call_data: Dict[str, Any]) -> Result:
    """
    Parseia dados de function call do formato OpenAI.
    Função pura de parsing.
    """
    try:
        tool_name = function_call_data.get("name")
        arguments_str = function_call_data.get("arguments", "{}")
        
        if not tool_name:
            return Result.error("Nome da função não fornecido")
        
        # Parsear argumentos JSON
        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError as e:
            return Result.error(f"Argumentos JSON inválidos: {str(e)}")
        
        # Criar ToolCall
        tool_call = create_tool_call(tool_name, arguments)
        return Result.ok(tool_call)
        
    except Exception as e:
        return Result.error(f"Erro ao parsear function call: {str(e)}")

def get_tools() -> Dict[str, Any]:
    """
    Função de compatibilidade com o código legado.
    Retorna ferramentas no formato esperado pelo agente antigo.
    """
    registry = get_tool_registry()
    return {
        "available_tools": registry.get_available_tools(),
        "executor": registry.executor,
        "registry": registry
    }

# ============================================================================
# SISTEMA DE PLUGINS (EXTENSIBILIDADE)
# ============================================================================

class ToolPlugin:
    """
    Classe base para plugins de ferramentas.
    Permite extensibilidade através de plugins externos.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.logger = get_logger()
    
    def register_tools(self, registry: ToolRegistry) -> Result:
        """
        Método a ser implementado por plugins.
        Deve registrar todas as ferramentas do plugin.
        """
        raise NotImplementedError("Plugin deve implementar register_tools()")
    
    def on_plugin_loaded(self) -> None:
        """Hook chamado após carregamento do plugin"""
        self.logger.log_info(f"Plugin carregado: {self.name} v{self.version}")
    
    def on_plugin_unloaded(self) -> None:
        """Hook chamado antes de descarregar o plugin"""
        self.logger.log_info(f"Plugin descarregado: {self.name}")

def load_plugin(plugin_class: Type[ToolPlugin]) -> Result:
    """
    Carrega um plugin de ferramentas.
    Função de integração para plugins externos.
    """
    try:
        plugin = plugin_class()
        registry = get_tool_registry()
        
        result = plugin.register_tools(registry)
        if result.success:
            plugin.on_plugin_loaded()
            return Result.ok(f"Plugin {plugin.name} carregado com sucesso")
        else:
            return Result.error(f"Falha ao carregar plugin {plugin.name}: {result.error}")
            
    except Exception as e:
        return Result.error(f"Erro ao carregar plugin: {str(e)}")
