from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
import json

# ============================================================================
# TIPOS LITERAIS E ENUMS
# ============================================================================

MessageRole = Literal["system", "user", "assistant", "function"]
EventType = Literal["user_message", "assistant_response", "function_call", "function_result", "error", "system"]
ToolCallStatus = Literal["pending", "success", "error"]

# ============================================================================
# MODELOS IMUTÁVEIS PRINCIPAIS
# ============================================================================

@dataclass(frozen=True)
class Message:
    """
    Representa uma mensagem imutável na conversa.
    Seguindo o padrão OpenAI ChatCompletion API.
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    function_call: Optional[Dict[str, Any]] = None
    name: Optional[str] = None  # Para mensagens de função
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário compatível com OpenAI API"""
        result = {
            "role": self.role,
            "content": self.content
        }
        if self.function_call:
            result["function_call"] = self.function_call
        if self.name:
            result["name"] = self.name
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Cria Message a partir de dicionário"""
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            function_call=data.get("function_call"),
            name=data.get("name")
        )

@dataclass(frozen=True)
class ToolCall:
    """
    Representa uma chamada de ferramenta/função imutável.
    """
    id: str
    name: str
    arguments: Dict[str, Any]
    status: ToolCallStatus = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa para persistência"""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """Deserializa da persistência"""
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data["arguments"],
            status=data["status"],
            result=data.get("result"),
            error=data.get("error"),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

@dataclass(frozen=True)
class Event:
    """
    Representa um evento imutável no sistema.
    Base para logging estruturado e rastreamento.
    """
    type: EventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    session_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Serializa evento para JSON (logging)"""
        return json.dumps({
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id
        }, ensure_ascii=False, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Event':
        """Deserializa evento do JSON"""
        data = json.loads(json_str)
        return cls(
            type=data["type"],
            data=data["data"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            session_id=data.get("session_id")
        )

@dataclass(frozen=True)
class Thread:
    """
    Representa uma thread de conversa imutável.
    Estado principal do agente - evolui através de eventos.
    """
    messages: List[Message] = field(default_factory=list)
    tools_calls: List[ToolCall] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: f"session_{datetime.utcnow().timestamp()}")
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_message(self, message: Message) -> 'Thread':
        """
        Retorna nova Thread com mensagem adicionada.
        Função pura - não modifica o estado atual.
        """
        return Thread(
            messages=self.messages + [message],
            tools_calls=self.tools_calls,
            session_id=self.session_id,
            created_at=self.created_at,
            updated_at=datetime.utcnow()
        )
    
    def add_tool_call(self, tool_call: ToolCall) -> 'Thread':
        """
        Retorna nova Thread com tool call adicionada.
        Função pura - não modifica o estado atual.
        """
        return Thread(
            messages=self.messages,
            tools_calls=self.tools_calls + [tool_call],
            session_id=self.session_id,
            created_at=self.created_at,
            updated_at=datetime.utcnow()
        )
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """
        Converte thread para formato compatível com OpenAI API.
        Função pura de transformação.
        """
        return [msg.to_dict() for msg in self.messages]
    
    def get_last_messages(self, n: int = 10) -> List[Message]:
        """
        Retorna as últimas N mensagens.
        Útil para contexto limitado.
        """
        return self.messages[-n:] if len(self.messages) > n else self.messages
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa thread completa"""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "tools_calls": [tc.to_dict() for tc in self.tools_calls],
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Thread':
        """Deserializa thread completa"""
        return cls(
            messages=[Message.from_dict(msg) for msg in data.get("messages", [])],
            tools_calls=[ToolCall.from_dict(tc) for tc in data.get("tools_calls", [])],
            session_id=data.get("session_id", f"session_{datetime.utcnow().timestamp()}"),
            created_at=datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.utcnow().isoformat()))
        )

# ============================================================================
# RESULTADO DE OPERAÇÕES (RESULT TYPES)
# ============================================================================

@dataclass(frozen=True)
class Result:
    """
    Tipo genérico para representar resultados de operações.
    Inspirado em Result<T, E> de linguagens funcionais.
    """
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    
    @classmethod
    def ok(cls, data: Any = None) -> 'Result':
        """Cria resultado de sucesso"""
        return cls(success=True, data=data)
    
    @classmethod
    def error(cls, error: str) -> 'Result':
        """Cria resultado de erro"""
        return cls(success=False, error=error)
    
    def map(self, func):
        """
        Aplica função aos dados se sucesso.
        Monadic pattern para composição funcional.
        """
        if self.success and self.data is not None:
            try:
                return Result.ok(func(self.data))
            except Exception as e:
                return Result.error(str(e))
        return self
    
    def flat_map(self, func):
        """
        FlatMap para Result monad.
        Permite composição de operações que retornam Result.
        """
        if self.success:
            try:
                return func(self.data)
            except Exception as e:
                return Result.error(str(e))
        return self
