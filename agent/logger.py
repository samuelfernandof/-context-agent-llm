import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import threading
from dataclasses import asdict

from models.models import Event, EventType, Result

# ============================================================================
# CONFIGURAÇÃO DE LOGGING ESTRUTURADO
# ============================================================================

class StructuredLogger:
    """
    Logger estruturado thread-safe para observabilidade do agente.
    Segue princípios funcionais e produz logs em formato JSON.
    """
    
    def __init__(self, log_file: str = "agent.log", level: int = logging.INFO):
        self.log_file = Path(log_file)
        self.level = level
        self._lock = threading.Lock()
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        """
        Configura o logger Python padrão para saída estruturada.
        Função pura de configuração.
        """
        # Criar diretório se não existir
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configurar logger
        self.logger = logging.getLogger("agent_logger")
        self.logger.setLevel(self.level)
        
        # Evitar duplicação de handlers
        if not self.logger.handlers:
            # Handler para arquivo
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(self.level)
            
            # Handler para console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)  # Só warnings/errors no console
            
            # Formato JSON para arquivo
            file_formatter = JsonFormatter()
            file_handler.setFormatter(file_formatter)
            
            # Formato simples para console
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def log_event(self, event: Event) -> Result:
        """
        Registra um evento estruturado.
        Função com efeito colateral controlado.
        """
        try:
            with self._lock:
                log_entry = {
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.type,
                    "session_id": event.session_id,
                    "data": event.data
                }
                
                # Log no nível apropriado baseado no tipo de evento
                level = self._get_log_level(event.type)
                self.logger.log(level, json.dumps(log_entry, ensure_ascii=False, default=str))
                
                return Result.ok(log_entry)
                
        except Exception as e:
            return Result.error(f"Erro ao registrar evento: {str(e)}")
    
    def _get_log_level(self, event_type: EventType) -> int:
        """
        Mapeia tipos de evento para níveis de log.
        Função pura de mapeamento.
        """
        level_mapping = {
            "error": logging.ERROR,
            "system": logging.WARNING,
            "function_call": logging.INFO,
            "function_result": logging.INFO,
            "user_message": logging.DEBUG,
            "assistant_response": logging.DEBUG
        }
        return level_mapping.get(event_type, logging.INFO)
    
    def log_info(self, message: str, **kwargs) -> Result:
        """Conveniência para logs de informação"""
        event = Event(
            type="system",
            data={"message": message, **kwargs}
        )
        return self.log_event(event)
    
    def log_error(self, error: str, **kwargs) -> Result:
        """Conveniência para logs de erro"""
        event = Event(
            type="error",
            data={"error": error, **kwargs}
        )
        return self.log_event(event)
    
    def log_function_call(self, function_name: str, arguments: Dict[str, Any], 
                         session_id: Optional[str] = None) -> Result:
        """Log específico para chamadas de função"""
        event = Event(
            type="function_call",
            data={
                "function_name": function_name,
                "arguments": arguments
            },
            session_id=session_id
        )
        return self.log_event(event)
    
    def log_function_result(self, function_name: str, result: Any, success: bool,
                           session_id: Optional[str] = None) -> Result:
        """Log específico para resultados de função"""
        event = Event(
            type="function_result",
            data={
                "function_name": function_name,
                "result": result,
                "success": success
            },
            session_id=session_id
        )
        return self.log_event(event)

class JsonFormatter(logging.Formatter):
    """
    Formatter personalizado para saída JSON estruturada.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Formata o log record como JSON estruturado.
        """
        # Tentar parsear como JSON se já estiver formatado
        try:
            # Se a mensagem já é JSON, usar diretamente
            message_data = json.loads(record.getMessage())
            log_entry = {
                "level": record.levelname,
                "logger": record.name,
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                **message_data
            }
        except (json.JSONDecodeError, ValueError):
            # Se não é JSON, criar estrutura básica
            log_entry = {
                "level": record.levelname,
                "logger": record.name,
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "message": record.getMessage()
            }
        
        # Adicionar informações de exceção se existirem
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)

# ============================================================================
# UTILITÁRIOS E CONTEXTO MANAGERS
# ============================================================================

@contextmanager
def log_operation(logger: StructuredLogger, operation_name: str, 
                 session_id: Optional[str] = None):
    """
    Context manager para log automático de operações.
    Registra início, fim e duração da operação.
    """
    start_time = datetime.utcnow()
    
    # Log de início
    logger.log_info(
        f"Iniciando operação: {operation_name}",
        operation=operation_name,
        session_id=session_id,
        start_time=start_time.isoformat()
    )
    
    try:
        yield
        
        # Log de sucesso
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        logger.log_info(
            f"Operação concluída: {operation_name}",
            operation=operation_name,
            session_id=session_id,
            duration_seconds=duration,
            status="success"
        )
        
    except Exception as e:
        # Log de erro
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        logger.log_error(
            f"Erro na operação: {operation_name}",
            operation=operation_name,
            session_id=session_id,
            duration_seconds=duration,
            error=str(e),
            status="error"
        )
        raise

def create_event(event_type: EventType, data: Dict[str, Any], 
                session_id: Optional[str] = None) -> Event:
    """
    Factory function para criar eventos de forma consistente.
    Função pura de criação.
    """
    return Event(
        type=event_type,
        data=data,
        session_id=session_id
    )

# ============================================================================
# LOG ANÁLISE E UTILITÁRIOS
# ============================================================================

class LogAnalyzer:
    """
    Analisador de logs para métricas e debugging.
    Funcionalidade read-only para análise de logs existentes.
    """
    
    def __init__(self, log_file: str = "agent.log"):
        self.log_file = Path(log_file)
    
    def read_events(self, session_id: Optional[str] = None) -> List[Event]:
        """
        Lê eventos do arquivo de log.
        Função pura de leitura.
        """
        events = []
        
        if not self.log_file.exists():
            return events
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        log_data = json.loads(line)
                        
                        # Filtrar por session_id se especificado
                        if session_id and log_data.get("session_id") != session_id:
                            continue
                        
                        # Converter para Event se possível
                        if "event_type" in log_data:
                            event = Event(
                                type=log_data["event_type"],
                                data=log_data.get("data", {}),
                                timestamp=datetime.fromisoformat(log_data["timestamp"]),
                                session_id=log_data.get("session_id")
                            )
                            events.append(event)
                            
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        # Log malformado, pular
                        continue
                        
        except Exception as e:
            # Erro de leitura, retornar lista vazia
            return []
        
        return events
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Gera resumo de uma sessão específica.
        Função pura de análise.
        """
        events = self.read_events(session_id)
        
        if not events:
            return {"session_id": session_id, "events_count": 0}
        
        # Análise básica
        event_types = {}
        for event in events:
            event_types[event.type] = event_types.get(event.type, 0) + 1
        
        return {
            "session_id": session_id,
            "events_count": len(events),
            "event_types": event_types,
            "start_time": events[0].timestamp.isoformat() if events else None,
            "end_time": events[-1].timestamp.isoformat() if events else None,
            "duration_minutes": (events[-1].timestamp - events[0].timestamp).total_seconds() / 60 if len(events) > 1 else 0
        }

# ============================================================================
# INSTÂNCIA GLOBAL (SINGLETON PATTERN)
# ============================================================================

# Logger global para uso em todo o sistema
_global_logger: Optional[StructuredLogger] = None

def get_logger() -> StructuredLogger:
    """
    Retorna instância global do logger (singleton pattern).
    Thread-safe lazy initialization.
    """
    global _global_logger
    
    if _global_logger is None:
        _global_logger = StructuredLogger()
    
    return _global_logger

# Funções de conveniência para uso direto
def log_event(event: Event) -> Result:
    """Função de conveniência para log de eventos"""
    return get_logger().log_event(event)

def log_info(message: str, **kwargs) -> Result:
    """Função de conveniência para log de informação"""
    return get_logger().log_info(message, **kwargs)

def log_error(error: str, **kwargs) -> Result:
    """Função de conveniência para log de erro"""
    return get_logger().log_error(error, **kwargs)
