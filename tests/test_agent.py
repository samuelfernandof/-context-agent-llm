"""
Tests Package - Testes do Sistema de Agente Funcional
=====================================================

Este pacote contém testes abrangentes para todos os componentes
do sistema de agente funcional, incluindo:

- Testes unitários para modelos imutáveis
- Testes de integração entre componentes
- Testes de sistema de memória persistente
- Testes de gerenciamento de contexto
- Testes de sistema de ferramentas
- Testes do loop principal do agente
- Testes de logging estruturado

Estrutura de testes:
- test_agent.py: Testes principais e abrangentes
- fixtures/: Dados de teste reutilizáveis
- integration/: Testes de integração específicos
- performance/: Testes de performance e stress

Como executar:
    # Todos os testes
    pytest tests/ -v
    
    # Apenas testes unitários
    pytest tests/ -v -m unit
    
    # Apenas testes de integração
    pytest tests/ -v -m integration
    
    # Com cobertura de código
    pytest tests/ --cov=agent --cov=models --cov-report=html
    
    # Testes paralelos
    pytest tests/ -v -n auto
"""

# ============================================================================
# CONFIGURAÇÃO GLOBAL DE TESTES
# ============================================================================

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Adicionar diretório raiz ao path para importações
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# UTILITÁRIOS DE TESTE COMPARTILHADOS
# ============================================================================

def create_test_db():
    """
    Cria banco de dados temporário para testes.
    
    Returns:
        str: Caminho para o banco temporário
    """
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    return path

def create_test_log():
    """
    Cria arquivo de log temporário para testes.
    
    Returns:
        str: Caminho para o arquivo de log temporário
    """
    fd, path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    return path

def cleanup_test_files(*paths):
    """
    Limpa arquivos de teste após uso.
    
    Args:
        *paths: Caminhos dos arquivos para limpar
    """
    for path in paths:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass

def create_test_directory():
    """
    Cria diretório temporário para testes.
    
    Returns:
        str: Caminho para o diretório temporário
    """
    return tempfile.mkdtemp()

def cleanup_test_directory(path):
    """
    Remove diretório de teste e todo seu conteúdo.
    
    Args:
        path: Caminho do diretório para remover
    """
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)

# ============================================================================
# DADOS DE TESTE PADRÃO
# ============================================================================

SAMPLE_MESSAGES = [
    {
        "role": "user",
        "content": "Olá, como você está?",
        "timestamp": "2023-01-01T10:00:30"
    },
    {
        "role": "assistant",
        "content": "Claro! Que tipo de cálculo você precisa fazer?",
        "timestamp": "2023-01-01T10:00:31"
    }
]

SAMPLE_TOOL_CALLS = [
    {
        "id": "call_123",
        "name": "calculate",
        "arguments": {"expression": "2 + 2"},
        "status": "success",
        "result": 4,
        "timestamp": "2023-01-01T10:01:00"
    },
    {
        "id": "call_456", 
        "name": "get_current_time",
        "arguments": {},
        "status": "success",
        "result": "2023-01-01 10:01:00 UTC",
        "timestamp": "2023-01-01T10:01:15"
    }
]

SAMPLE_EVENTS = [
    {
        "type": "user_message",
        "data": {"content": "Olá!", "user_id": "test_user"},
        "timestamp": "2023-01-01T10:00:00",
        "session_id": "test_session"
    },
    {
        "type": "function_call",
        "data": {"function_name": "calculate", "arguments": {"x": 5, "y": 10}},
        "timestamp": "2023-01-01T10:00:30",
        "session_id": "test_session"
    },
    {
        "type": "error",
        "data": {"error": "Conexão perdida", "code": 500},
        "timestamp": "2023-01-01T10:01:00",
        "session_id": "test_session"
    }
]

SAMPLE_CONFIG = {
    "model": "mistralai/mistral-7b-instruct",
    "api_key": "test-key-123",
    "temperature": 0.7,
    "max_tokens": 2000,
    "enable_memory_persistence": True,
    "enable_function_calling": True,
    "max_retries": 3,
    "context_strategy": "default"
}

# ============================================================================
# MOCKS E FIXTURES GLOBAIS
# ============================================================================

class MockOpenAIResponse:
    """
    Mock para respostas da API OpenAI.
    Simula diferentes tipos de resposta para testes.
    """
    
    @staticmethod
    def simple_text_response(content="Resposta de teste"):
        """Resposta simples de texto"""
        return {
            'choices': [{
                'message': {
                    'content': content,
                    'function_call': None
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': 100,
                'completion_tokens': 20,
                'total_tokens': 120
            }
        }
    
    @staticmethod
    def function_call_response(function_name, arguments):
        """Resposta com function call"""
        return {
            'choices': [{
                'message': {
                    'content': '',
                    'function_call': {
                        'name': function_name,
                        'arguments': str(arguments) if isinstance(arguments, dict) else arguments
                    }
                },
                'finish_reason': 'function_call'
            }],
            'usage': {
                'prompt_tokens': 150,
                'completion_tokens': 30,
                'total_tokens': 180
            }
        }
    
    @staticmethod
    def error_response():
        """Resposta de erro"""
        raise Exception("API Error: Rate limit exceeded")

class MockDatabase:
    """
    Mock para banco de dados SQLite.
    Útil para testes que não precisam de persistência real.
    """
    
    def __init__(self):
        self.data = {}
        self.messages = []
        self.events = []
    
    def save_thread(self, thread_data):
        """Simula salvamento de thread"""
        session_id = thread_data.get('session_id', 'unknown')
        self.data[session_id] = thread_data
        return True
    
    def load_thread(self, session_id):
        """Simula carregamento de thread"""
        return self.data.get(session_id)
    
    def list_sessions(self):
        """Simula listagem de sessões"""
        return list(self.data.keys())

# ============================================================================
# VALIDADORES DE TESTE
# ============================================================================

def validate_message_structure(message_dict):
    """
    Valida estrutura de mensagem em testes.
    
    Args:
        message_dict: Dicionário com dados da mensagem
    
    Returns:
        bool: True se estrutura está correta
    """
    required_fields = ['role', 'content', 'timestamp']
    
    for field in required_fields:
        if field not in message_dict:
            return False
    
    valid_roles = ['system', 'user', 'assistant', 'function']
    if message_dict['role'] not in valid_roles:
        return False
    
    return True

def validate_thread_structure(thread_dict):
    """
    Valida estrutura de thread em testes.
    
    Args:
        thread_dict: Dicionário com dados da thread
    
    Returns:
        bool: True se estrutura está correta
    """
    required_fields = ['session_id', 'messages', 'created_at', 'updated_at']
    
    for field in required_fields:
        if field not in thread_dict:
            return False
    
    # Validar mensagens
    if not isinstance(thread_dict['messages'], list):
        return False
    
    for message in thread_dict['messages']:
        if not validate_message_structure(message):
            return False
    
    return True

def validate_tool_call_structure(tool_call_dict):
    """
    Valida estrutura de tool call em testes.
    
    Args:
        tool_call_dict: Dicionário com dados do tool call
    
    Returns:
        bool: True se estrutura está correta
    """
    required_fields = ['id', 'name', 'arguments', 'status']
    
    for field in required_fields:
        if field not in tool_call_dict:
            return False
    
    valid_statuses = ['pending', 'success', 'error']
    if tool_call_dict['status'] not in valid_statuses:
        return False
    
    return True

# ============================================================================
# GERADORES DE DADOS DE TESTE
# ============================================================================

def generate_test_messages(count=5, start_time=None):
    """
    Gera mensagens de teste.
    
    Args:
        count: Número de mensagens a gerar
        start_time: Timestamp inicial (opcional)
    
    Returns:
        list: Lista de dicionários de mensagens
    """
    from datetime import datetime, timedelta
    
    if start_time is None:
        start_time = datetime.utcnow()
    
    messages = []
    roles = ['user', 'assistant']
    
    for i in range(count):
        role = roles[i % 2]  # Alternar entre user e assistant
        content = f"Mensagem de teste #{i+1} do {role}"
        timestamp = start_time + timedelta(seconds=i*30)
        
        messages.append({
            'role': role,
            'content': content,
            'timestamp': timestamp.isoformat()
        })
    
    return messages

def generate_test_thread(session_id=None, message_count=4):
    """
    Gera thread de teste completa.
    
    Args:
        session_id: ID da sessão (opcional)
        message_count: Número de mensagens
    
    Returns:
        dict: Dados da thread de teste
    """
    from datetime import datetime
    
    if session_id is None:
        session_id = f"test_session_{datetime.utcnow().timestamp()}"
    
    created_at = datetime.utcnow()
    messages = generate_test_messages(message_count, created_at)
    
    return {
        'session_id': session_id,
        'messages': messages,
        'tools_calls': [],
        'created_at': created_at.isoformat(),
        'updated_at': (created_at + timedelta(minutes=5)).isoformat()
    }

def generate_test_config(**overrides):
    """
    Gera configuração de teste.
    
    Args:
        **overrides: Valores para sobrescrever padrões
    
    Returns:
        dict: Configuração de teste
    """
    config = SAMPLE_CONFIG.copy()
    config.update(overrides)
    return config

# ============================================================================
# COMPARADORES DE TESTE
# ============================================================================

def compare_messages(msg1, msg2, ignore_timestamp=True):
    """
    Compara duas mensagens para testes.
    
    Args:
        msg1: Primeira mensagem
        msg2: Segunda mensagem
        ignore_timestamp: Se deve ignorar timestamps
    
    Returns:
        bool: True se mensagens são equivalentes
    """
    fields_to_compare = ['role', 'content']
    if not ignore_timestamp:
        fields_to_compare.append('timestamp')
    
    for field in fields_to_compare:
        if msg1.get(field) != msg2.get(field):
            return False
    
    return True

def compare_threads(thread1, thread2, ignore_timestamps=True):
    """
    Compara duas threads para testes.
    
    Args:
        thread1: Primeira thread
        thread2: Segunda thread
        ignore_timestamps: Se deve ignorar timestamps
    
    Returns:
        bool: True se threads são equivalentes
    """
    # Comparar metadados básicos
    if thread1.get('session_id') != thread2.get('session_id'):
        return False
    
    # Comparar número de mensagens
    messages1 = thread1.get('messages', [])
    messages2 = thread2.get('messages', [])
    
    if len(messages1) != len(messages2):
        return False
    
    # Comparar mensagens individuais
    for m1, m2 in zip(messages1, messages2):
        if not compare_messages(m1, m2, ignore_timestamps):
            return False
    
    return True

# ============================================================================
# UTILITÁRIOS DE ASSERÇÃO PERSONALIZADOS
# ============================================================================

class TestAssertions:
    """
    Asserções personalizadas para testes do agente.
    """
    
    @staticmethod
    def assert_valid_message(message_dict):
        """Asserta que mensagem tem estrutura válida"""
        assert validate_message_structure(message_dict), f"Estrutura de mensagem inválida: {message_dict}"
    
    @staticmethod
    def assert_valid_thread(thread_dict):
        """Asserta que thread tem estrutura válida"""
        assert validate_thread_structure(thread_dict), f"Estrutura de thread inválida: {thread_dict}"
    
    @staticmethod
    def assert_valid_tool_call(tool_call_dict):
        """Asserta que tool call tem estrutura válida"""
        assert validate_tool_call_structure(tool_call_dict), f"Estrutura de tool call inválida: {tool_call_dict}"
    
    @staticmethod
    def assert_result_success(result, message="Operação deveria ter sucesso"):
        """Asserta que Result indica sucesso"""
        assert result.success, f"{message}. Erro: {result.error}"
    
    @staticmethod
    def assert_result_error(result, message="Operação deveria falhar"):
        """Asserta que Result indica erro"""
        assert not result.success, f"{message}. Dados: {result.data}"
    
    @staticmethod
    def assert_messages_equal(msg1, msg2, ignore_timestamp=True):
        """Asserta que duas mensagens são equivalentes"""
        assert compare_messages(msg1, msg2, ignore_timestamp), f"Mensagens não são equivalentes: {msg1} != {msg2}"
    
    @staticmethod
    def assert_threads_equal(thread1, thread2, ignore_timestamps=True):
        """Asserta que duas threads são equivalentes"""
        assert compare_threads(thread1, thread2, ignore_timestamps), f"Threads não são equivalentes"

# ============================================================================
# CONFIGURAÇÃO DE AMBIENTE DE TESTE
# ============================================================================

def setup_test_environment():
    """
    Configura ambiente para execução de testes.
    """
    # Configurar variáveis de ambiente para testes
    os.environ.update({
        'OPENROUTER_API_KEY': 'test-key-for-testing',
        'AGENT_MODEL': 'test-model',
        'AGENT_TEMPERATURE': '0.0',  # Determinístico para testes
        'AGENT_LOG_LEVEL': 'DEBUG',
        'AGENT_MEMORY_DB': ':memory:',  # SQLite em memória
        'AGENT_MAX_RETRIES': '1',  # Falhar rápido em testes
        'ENVIRONMENT': 'testing'
    })

def teardown_test_environment():
    """
    Limpa ambiente após execução de testes.
    """
    # Remover variáveis de teste
    test_vars = [
        'OPENROUTER_API_KEY', 'AGENT_MODEL', 'AGENT_TEMPERATURE',
        'AGENT_LOG_LEVEL', 'AGENT_MEMORY_DB', 'AGENT_MAX_RETRIES',
        'ENVIRONMENT'
    ]
    
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]

# ============================================================================
# SKIP CONDITIONS PARA TESTES
# ============================================================================

def skip_if_no_api_key():
    """
    Pula teste se não há API key disponível.
    Útil para testes que requerem API real.
    """
    import pytest
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key.startswith('test-'):
        return pytest.mark.skip(reason="API key real não disponível")
    
    return pytest.mark.api

def skip_if_no_internet():
    """
    Pula teste se não há conexão com internet.
    """
    import pytest
    import socket
    
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return pytest.mark.online
    except OSError:
        return pytest.mark.skip(reason="Conexão com internet não disponível")

# ============================================================================
# EXPORTAÇÕES PÚBLICAS
# ============================================================================

__all__ = [
    # Utilitários de arquivo
    'create_test_db',
    'create_test_log', 
    'cleanup_test_files',
    'create_test_directory',
    'cleanup_test_directory',
    
    # Dados de teste
    'SAMPLE_MESSAGES',
    'SAMPLE_TOOL_CALLS',
    'SAMPLE_EVENTS',
    'SAMPLE_CONFIG',
    
    # Mocks
    'MockOpenAIResponse',
    'MockDatabase',
    
    # Validadores
    'validate_message_structure',
    'validate_thread_structure',
    'validate_tool_call_structure',
    
    # Geradores
    'generate_test_messages',
    'generate_test_thread',
    'generate_test_config',
    
    # Comparadores
    'compare_messages',
    'compare_threads',
    
    # Asserções
    'TestAssertions',
    
    # Ambiente
    'setup_test_environment',
    'teardown_test_environment',
    
    # Skip conditions
    'skip_if_no_api_key',
    'skip_if_no_internet'
]

# ============================================================================
# INICIALIZAÇÃO AUTOMÁTICA
# ============================================================================

# Configurar ambiente de teste automaticamente quando módulo é importado
try:
    setup_test_environment()
except Exception:
    # Falha silenciosa se não conseguir configurar
    pass00:00"
    },
    {
        "role": "assistant", 
        "content": "Estou bem, obrigado! Como posso ajudar você hoje?",
        "timestamp": "2023-01-01T10:00:01"
    },
    {
        "role": "user",
        "content": "Preciso de ajuda com um cálculo matemático.",
        "timestamp": "2023-01-01T10:
